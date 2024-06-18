from typing import Dict, List

import argilla as rg
import pandas as pd
import pandera as pa


def pandera_schema_to_argilla_dataset(
        schema: pa.DataFrameSchema,
        papers: pd.DataFrame,
        fields: List[rg.TextField],
        vectors_settings: List[rg.VectorSettings] = None,
        **kwargs) -> rg.FeedbackDataset:
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

    return rg.FeedbackDataset(
        fields=fields,
        questions=questions,
        metadata_properties=list(metadata_properties.values()) if metadata_properties else None,
        guidelines=schema.description,
        vectors_settings=vectors_settings,
        **kwargs
    )
