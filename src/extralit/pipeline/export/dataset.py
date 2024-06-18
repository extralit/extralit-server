from typing import Dict, List, Optional, Literal, Union, Any

import argilla as rg
import pandas as pd
import pandera as pa


def create_papers_dataset(
        schema: pa.DataFrameSchema,
        papers: pd.DataFrame,
        fields: List[rg.TextField] = None,
        vectors_settings: List[rg.VectorSettings] = None,
        **kwargs) -> rg.FeedbackDataset:
    fields = fields or []
    questions = []
    metadata_properties = {}

    for field_name, column in schema.columns.items():
        if field_name == "reference":
            metadata_properties[field_name] = rg.TermsMetadataProperty(name=field_name, title=field_name.capitalize(), visible_for_annotators=True)
            continue

        if column.dtype.type == bool:
            question = rg.LabelQuestion(
                name=field_name.lower(), title=column.title or field_name, description=column.description,
                labels={'True': 'YES', 'False': 'NO'}, required=not column.nullable)
        elif column.dtype.type == list:
            labels = next((check.statistics['isin'] for check in column.checks if 'isin' in check.statistics), None)
            question = rg.MultiLabelQuestion(
                name=field_name.lower(), title=column.title or field_name, description=column.description,
                labels=labels, required=not column.nullable)
        else:
            question = rg.TextQuestion(
                name=field_name.lower(), title=column.title or field_name, description=column.description,
                required=not column.nullable, use_markdown=True)

        questions.append(question)

    for column_name, dtype in papers.dtypes.items():
        if column_name in schema.columns or column_name=="file_path": continue
        if dtype == bool:
            metadata_prop = rg.TermsMetadataProperty(name=column_name, title=column_name.capitalize(), visible_for_annotators=True)
        elif dtype == float:
            metadata_prop = rg.FloatMetadataProperty(name=column_name, title=column_name.capitalize(), visible_for_annotators=True)
        elif dtype == int:
            metadata_prop = rg.IntegerMetadataProperty(name=column_name, title=column_name.capitalize(), visible_for_annotators=True)
        elif dtype == object:
            metadata_prop = rg.TermsMetadataProperty(name=column_name, title=column_name.capitalize(), visible_for_annotators=True)
        else:
            metadata_prop = rg.TermsMetadataProperty(name=column_name, title=column_name.capitalize(), visible_for_annotators=True)

        metadata_properties[column_name] = metadata_prop

    if not any(field.name == 'metadata' for field in fields):
        fields.insert(0, rg.TextField(name="metadata", title="Metadata", use_markdown=True))

    return rg.FeedbackDataset(
        fields=fields,
        questions=questions,
        metadata_properties=list(metadata_properties.values()) if metadata_properties else None,
        guidelines=schema.description,
        vectors_settings=vectors_settings,
        **kwargs
    )


def create_extraction_dataset(
        fields: Optional[List[rg.TextField]]=None,
        questions: Optional[List[rg.TextQuestion]]=None,
        metadata_properties: Optional[List[rg.TermsMetadataProperty]] = None,
        vectors_settings: Optional[List[rg.VectorSettings]]=None) -> rg.FeedbackDataset:
    extraction_dataset = rg.FeedbackDataset(
        guidelines="Manually validate every data entries in the data extraction sheet to build a "
                   "gold-standard validation dataset.",
        fields=[
            rg.TextField(name="metadata", title="Reference:", required=True, use_markdown=True),
            rg.TextField(name="extraction", title="Extracted data:", required=True, use_table=True),
            rg.TextField(name="context", title="Top relevant segments:", required=False, use_markdown=True),
            *(fields or [])
        ],
        questions=[
            rg.MultiLabelQuestion(
                name="context-relevant",
                title="Select the segment ID(s) attributed this extraction:",
                description="Note that the table number isn't the same as the segment ID. Please refer to Top relevant segments on the left hand-side.",
                labels=list(range(50)),
                visible_labels=3,
                required=True,
            ),

            rg.MultiLabelQuestion(
                name="extraction-source",
                title="Which source data type(s) did the extracted data primarily came from?",
                labels=["Text", "Table", "Figure"],
                required=True,
            ),

            rg.TextQuestion(
                name="extraction-correction",
                title="Provide a correction to the extracted data:",
                required=False,
                use_table=True,
            ),
            rg.TextQuestion(
                name="notes",
                title="Note any special case here or justify decisions made in the extraction:",
                required=False,
                use_markdown=True,
            ),
            *(questions or [])
        ],
        vectors_settings=vectors_settings,
        metadata_properties=[
            rg.TermsMetadataProperty(
                name="reference",
                title="Reference",
                visible_for_annotators=True),
            rg.TermsMetadataProperty(
                name="type",
                title="Question Type",
                visible_for_annotators=True),
            *(metadata_properties or [])
        ],
    )

    return extraction_dataset