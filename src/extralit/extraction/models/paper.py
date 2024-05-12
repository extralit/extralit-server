import itertools
from typing import Dict, Iterator, Tuple, Optional, Union

import pandas as pd
import pandera as pa
from extralit.extraction.models.schema import SchemaStructure
from pandera.api.base.model import MetaModel
from pydantic.v1 import BaseModel, Field

from extralit.schema.checks import register_check_methods

register_check_methods()


class PaperExtraction(BaseModel):
    reference: str
    extractions: Dict[str, pd.DataFrame] = Field(default_factory=dict)
    schemas: SchemaStructure = Field(default_factory=SchemaStructure)
    durations: Dict[str, Optional[float]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def get_joined_data(self, schema_name: str, drop_joined_index=False) -> pd.DataFrame:
        schema = self.schemas[schema_name]

        df = self[schema_name]
        if not isinstance(schema, pa.DataFrameSchema) and hasattr(schema, 'to_schema'):
            schema = schema.to_schema()
        else:
            assert isinstance(schema,
                              pa.DataFrameSchema), f"Expected DataFrameModel or DataFrameSchema, got {type(schema)}"

        # For each '_ref' key, find the matching DataFrame with the same DataFrameModel prefix
        for ref_index_name in schema.index.names:
            ref_schema = ref_index_name.rsplit('_ref', 1)[0].lower()

            matching_df = next((value for key, value in self.extractions.items() if str(key).lower() == ref_schema),
                               None)
            if matching_df is not None:
                try:
                    df = df.join(matching_df.rename_axis(index={'reference': ref_index_name}), how='left',
                                 rsuffix='_joined')
                    df = overwrite_joined_columns(df, rsuffix='_joined')
                    if drop_joined_index and ref_index_name in df.index.names:
                        df = df.reset_index(level=ref_index_name, drop=True)
                except Exception as e:
                    print(f"Failed to join `{ref_schema}` to {schema.name}: {e}",
                          # df.shape, matching_df.shape, df.index.names, matching_df.index.names, df.columns, matching_df.columns
                          )
                    raise e

        return df

    def __getitem__(self, item: str) -> pd.DataFrame:
        if isinstance(item, pa.DataFrameSchema):
            return self.extractions[item.name]
        elif isinstance(item, MetaModel):
            return self.extractions[str(item)]
        return self.extractions[item]

    def __contains__(self, item: Union[pa.DataFrameModel, str]) -> bool:
        if isinstance(item, pa.DataFrameSchema):
            return item.name in self.extractions
        elif isinstance(item, MetaModel):
            return str(item) in self.extractions
        return item in self.extractions

    def __getattr__(self, item: str) -> pd.DataFrame:
        if self.__contains__(item):
            return self.__getitem__(item)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __dir__(self) -> Iterator[str]:
        extraction_keys = [str(key) for key in self.extractions.keys()]
        return itertools.chain(super().__dir__(), extraction_keys)

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        assert isinstance(key, str), f"Expected str, got {type(key)}"
        self.extractions[key] = value

    def items(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        return self.extractions.items()

    def __repr_args__(self):
        args = [(k, v.dropna(axis=1, how='all').shape) for k, v in self.extractions.items() if v.size]
        return args


def overwrite_joined_columns(df: pd.DataFrame, rsuffix='_joined') -> pd.DataFrame:
    # Find all columns with the '_joined' suffix
    joined_columns = [col for col in df.columns if col.endswith(rsuffix)]

    for joined_col in joined_columns:
        # Get the original column name by removing the '_joined' suffix
        original_col = joined_col.rsplit(rsuffix, 1)[0]

        # Overwrite the original column with the '_joined' column
        df[original_col] = df[joined_col]

        # Drop the '_joined' column
        df = df.drop(columns=joined_col)

    return df
