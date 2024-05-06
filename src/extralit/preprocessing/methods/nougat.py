import gc
import logging
import re
from functools import partial
from pathlib import Path
from typing import List

import pypandoc
import pypdf
import pypdfium2
import torch
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from nougat import NougatModel
from nougat.dataset.rasterize import rasterize_paper
from nougat.postprocessing import close_envs, markdown_compatible
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import ImageDataset, LazyDataset
from nougat.utils.device import default_batch_size, move_to_device
from pydantic import BaseModel
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from extralit.convert.html_table import remove_html_styles
from extralit.convert.text import remove_longest_repeated_subsequence
from extralit.convert.text import remove_markdown_from_string
from extralit.preprocessing.segment import TableSegment, TextSegment, Segments


class NougatOCR:
    def __init__(self, model_tag='0.1.0-base', full_precision=False, markdown=True, skipping=True):
        model_path = get_checkpoint(model_tag=model_tag)
        self.model: NougatModel = NougatModel.from_pretrained(model_path)
        self.markdown = markdown
        self.skipping = skipping

        self.batch_size = default_batch_size()
        self.model = move_to_device(self.model, bf16=not full_precision, cuda=self.batch_size > 0)

        if self.batch_size <= 0:
            self.batch_size = 1

        self.model.eval()

    def batch_predict(self, file_paths: List[Path]) -> List[List[str]]:
        datasets = []

        for pdf in file_paths:
            if not pdf.exists():
                continue

            try:
                dataset = LazyDataset(
                    pdf,
                    partial(self.model.encoder.prepare_input, random_padding=False),
                )
            except pypdf.errors.PdfStreamError:
                logging.info(f"Could not load file {str(pdf)}.")
                continue
            datasets.append(dataset)
        if len(datasets) == 0:
            return

        dataloader = torch.utils.data.DataLoader(
            ConcatDataset(datasets),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate,
        )

        documents = []
        predictions = []
        file_index = 0
        page_num = 0
        for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
            model_output = self.model.inference(
                image_tensors=sample, early_stopping=self.skipping
            )
            # check if itnrecal output is faulty
            for j, output in enumerate(model_output["predictions"]):
                if page_num == 0:
                    logging.info(
                        "Processing file %s with %i pages"
                        % (datasets[file_index].name, datasets[file_index].size)
                    )
                page_num += 1
                if output.strip() == "[MISSING_PAGE_POST]":
                    # uncaught repetitions -- most likely empty page
                    predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
                elif self.skipping and model_output["repeats"][j] is not None:
                    if model_output["repeats"][j] > 0:
                        # If we end up here, it means the output is most likely not complete and was truncated.
                        logging.warning(f"Skipping page {page_num} due to repetitions.")
                        predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                    else:
                        # If we end up here, it means the document page is too different from the training domain.
                        # This can happen e.g. for cover pages.
                        predictions.append(
                            f"\n\n[MISSING_PAGE_EMPTY:{i * self.batchsize + j + 1}]\n\n"
                        )
                else:
                    if self.markdown:
                        output = markdown_compatible(output)
                    predictions.append(output)

                if is_last_page[j]:
                    documents.append(predictions)

                    predictions = []
                    page_num = 0
                    file_index += 1

                    # clear the torch cache and memory
                    self.empty_cache()

        return documents

    def predict(self, file_path: str, verbose=True) -> List[str]:
        with open(file_path, 'rb') as file:
            pdfbin = file.read()
            pdf = pypdfium2.PdfDocument(pdfbin)
            pages = list(range(len(pdf)))

        compute_pages = pages.copy()
        images = rasterize_paper(pdf, pages=compute_pages)

        dataset = ImageDataset(
            images,
            partial(self.model.encoder.prepare_input, random_padding=False),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )

        # clear the torch cache and memory
        self.empty_cache()

        predictions = [""] * len(pages)
        for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader), disable=not verbose):
            if sample is None:
                continue

            model_output = self.model.inference(image_tensors=sample, early_stopping=self.skipping)

            for page_idx, output in enumerate(model_output["predictions"]):
                if model_output["repeats"][page_idx] is not None:
                    if model_output["repeats"][page_idx] > 0:
                        disclaimer = "\n\n%s\n\n"
                    else:
                        disclaimer = "\n\n%s\n\n"

                    rest = close_envs(model_output["repetitions"][page_idx]).strip()
                    if len(rest) > 0:
                        disclaimer = disclaimer % rest
                    else:
                        disclaimer = ""
                else:
                    disclaimer = ""

                predictions[pages.index(compute_pages[idx * self.batch_size + page_idx])] = \
                    markdown_compatible(output) + disclaimer

            self.empty_cache()
            gc.collect(generation=2)
        return predictions

    def empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def process_outputs(self, model_output):
        predictions = [""] * len(model_output['predictions'])
        for page_idx, output in enumerate(model_output["predictions"]):
            if model_output["repeats"][page_idx] is not None:
                if model_output["repeats"][page_idx] > 0:
                    disclaimer = "\n\n+++ ==WARNING: Truncated because of repetitions==\n%s\n+++\n\n"
                else:
                    disclaimer = "\n\n+++ ==ERROR: No output for this page==\n%s\n+++\n\n"

                rest = close_envs(model_output["repetitions"][page_idx]).strip()
                if len(rest) > 0:
                    disclaimer = disclaimer % rest
                else:
                    disclaimer = ""
            else:
                disclaimer = ""

            predictions[page_idx] = markdown_compatible(output) + disclaimer
        return predictions


class NougatOutput(BaseModel):
    reference: str
    pages: List[str]


def get_text_segments(pages: List[str], title="Title") -> Segments:
    segments = Segments()
    current_segment = None
    stored_header = ""
    parents_stack = []

    for page_number, page in enumerate(pages, start=1):
        page = remove_longest_repeated_subsequence(page, min_substring_len=1, min_repeats=10)
        page = re.sub(r'\n*\\begin{table}.*?\\end{table}\n.*?(\n|$)', '', page, flags=re.DOTALL)
        page = re.sub(r'\n*\\begin{tabular}.*?\\end{tabular}\n.*?(\n|$)', '', page, flags=re.DOTALL)
        if not current_segment and page_number == 1:
            current_segment = TextSegment(header=title, level=1, page_number=page_number, text='')

        for line in page.split('\n'):
            header_match = re.match(r'(#+)\s*(.*)', line)
            if header_match:
                if current_segment and (current_segment.text or current_segment.relationships):
                    segments.items.append(current_segment)
                level = len(header_match.group(1))
                while parents_stack and parents_stack[-1].level >= level:
                    parents_stack.pop()
                parent = parents_stack[-1] if parents_stack else None
                current_segment = TextSegment(header=f"{stored_header}{header_match.group(2)}",
                                              level=level,
                                              page_number=page_number,
                                              text='')

                if parent:
                    current_segment.relationships[NodeRelationship.PARENT] = \
                        RelatedNodeInfo(node_id=parent.id, )

                    parent.relationships.setdefault(NodeRelationship.CHILD, []).append(
                        RelatedNodeInfo(node_id=current_segment.id, )
                    )
                stored_header = ""
                parents_stack.append(current_segment)

            elif current_segment:
                current_segment.text += line + '\n'

    segments.make_headers_unique()
    return segments


def correct_column_definition(latex_table: str) -> str:
    # Split the table into rows
    rows = re.split(r'\\', latex_table)

    # Find the row with the maximum number of columns
    max_columns = max(row.count('&') for row in rows if r'\begin' not in row and r'\end' not in row)

    # Generate the corrected column definition
    corrected_definition = ' '.join(['c' for _ in range(max_columns + 1)])

    # Replace the original column definition in the \begin{tabular} line
    corrected_table = re.sub(r'\\begin{tabular}{.*?}', r'\\begin{tabular}{' + corrected_definition + '}', latex_table)

    return corrected_table


def get_table_segments(pages: List[str]) -> Segments:
    segments = Segments()

    for page_number, page_text in enumerate(pages, start=1):
        # Regular expression pattern for LaTeX tables
        pattern = r'(\\begin{table}.*?\\end{tabular}(.*?)\\end{table})\n(.*?)(\n|$)'
        matches = re.findall(pattern, page_text, re.DOTALL)
        if not matches:
            pattern = r'(\\begin{tabular}.*?\\end{tabular}(.*?))\n(.*?)(\n|$)'
            matches = re.findall(pattern, page_text, re.DOTALL)

        for match in matches:
            table_content, footer, caption, *_ = match
            table_html, table_markdown = '', ''

            try:
                table_html = pypandoc.convert_text(table_content, 'html', format='latex')
            except Exception as e:
                try:
                    table_content = correct_column_definition(table_content)
                    table_html = pypandoc.convert_text(table_content, 'html', format='latex')
                except Exception as e:
                    logging.warning(f"Could not convert table to HTML: {e.__str__()}")
            finally:
                if table_html:
                    table_html = remove_html_styles(table_html)
                    table_html = remove_markdown_from_string(table_html)
                else:
                    continue

            if not table_html.startswith(r'<table>'):
                continue
            else:
                table_html = table_html.replace('—', '-').replace('·', '.')

            caption = caption.strip() + (('\n' + footer.strip()) if footer else '')

            # Create a Segment object
            segment = TableSegment(
                header=caption,
                text=table_markdown,
                html=table_html,
                page_number=page_number,
                original=table_content,
                source='nougat',
            )
            segments.items.append(segment)

    return segments
