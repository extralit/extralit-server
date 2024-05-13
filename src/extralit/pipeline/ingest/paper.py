from typing import Dict, Type, List, Literal, Optional

import argilla as rg
import pandas as pd

from extralit.convert.json_table import json_to_df, is_json_table
from extralit.extraction.models.paper import PaperExtraction
from extralit.extraction.models.schema import SchemaStructure
from extralit.pipeline.ingest.record import get_record_data
from extralit.preprocessing.segment import Segments, TableSegment, FigureSegment


def get_paper_tables(paper: pd.Series,
                     dataset: rg.FeedbackDataset,
                     select: str = 'text-correction',
                     response_status: List[Literal['discarded', 'submitted', 'pending', 'draft']] = [
                         'submitted']) -> Segments:
    """
    Get the tables manually annotated a given paper in an Argilla (Preprocessing) FeedbackDataset.

    Args:
        paper: pd.Series, required
            A paper from the dataset.
        dataset: rg.FeedbackDataset, required
            The Argilla (Preprocessing) FeedbackDataset.
        select: str, default='text-correction'
            The field to select from the dataset records.
        response_status: List[str], default=['discarded']

    Returns:
        Segments: The tables manually annotated for the given paper.
    """
    records = dataset.filter_by(
        metadata_filters=rg.TermsMetadataFilter(
            name='reference',
            values=[paper.name]),
        response_status=response_status).records

    segments = Segments()
    for record in records:
        print(record.id, record.metadata)
        values = get_record_data(record,
                                 fields=['text-1', 'text-2', 'text-3', 'text-4', 'text-5', 'header'],
                                 answers=['text-correction', 'header-correction', 'footer-correction', 'ranking',
                                          'duration'],
                                 metadatas=['page_number', 'number'],
                                 skip_status=['discarded'])
        if 'ranking' not in values or values['ranking'] == 'none':
            continue

        try:
            if select.strip().lower() == 'ranking' and values['ranking'] in values:
                html = values[values['ranking']]

            elif select in values and 'correction' in select and not values[select]:
                # Skip the empty corrections
                html = values[values['ranking']]

            elif select in values:
                html = values[select]
            else:
                continue
        except:
            continue

        if 'header-correction' in values:
            header = values['header-correction']
        else:
            header = values['header']

        type = values.get('number', 'table').split(' ')[0].lower()
        if type == 'figure':
            segment = FigureSegment(
                id=record.id,
                header=header.strip(),
                footer=values.get('footer-correction', None),
                page_number=values.get('page_number', None),
                text=html,
                html=html,
                duration=values.get('duration', None),
            )
        else:
            segment = TableSegment(
                id=record.id,
                header=header.strip(),
                footer=values.get('footer-correction', None),
                page_number=values.get('page_number', None),
                text=html,
                html=html,
                duration=values.get('duration', None),
            )

        segments.items.append(segment)

    return segments


def get_paper_extractions(paper: pd.Series, dataset: rg.FeedbackDataset,
                          schemas: SchemaStructure,
                          field='extraction', answer='extraction-correction',
                          user: Optional[rg.User] = None, ) -> PaperExtraction:

    records = dataset.filter_by(
        metadata_filters=rg.TermsMetadataFilter(
            name='reference',
            values=[paper.name])).records

    extractions = {}
    durations = {}
    for record in records:
        if record.metadata['reference'] != paper.name:
            continue

        outputs = get_record_data(record,
                                  fields=field,
                                  answers=[answer, 'duration'] if answer else ['duration'],
                                  users=user)
        if answer in outputs and is_json_table(outputs[answer]):
            table_json = outputs[answer]
        elif field in outputs and is_json_table(outputs[field]):
            table_json = outputs[field]
        else:
            table_json = None

        duration = outputs.get('duration', None)

        for schema in schemas.schemas:
            if schema.name == record.metadata['type']:
                extractions[schema.name] = json_to_df(table_json, schema=schema)
                durations[schema.name] = duration

    return PaperExtraction(extractions=extractions, schemas=schemas, durations=durations)

