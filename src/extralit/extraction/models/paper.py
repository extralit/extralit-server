import itertools
import logging
import os
from collections import deque
from glob import glob
from typing import Dict, Iterator, Tuple, Optional, Union, List

import pandas as pd
import pandera as pa
from pandera.api.base.model import MetaModel
from pandera.io import from_yaml, from_json
from pydantic.v1 import BaseModel, Field, validator

from extralit.schema.checks import register_check_methods

register_check_methods()

def topological_sort(schema_name: str, visited: Dict[str, int], stack: deque,
                     dependencies: Dict[str, List[str]]) -> None:
    visited[schema_name] = 1  # Gray

    for i in dependencies.get(schema_name, []):
        if visited[i] == 1:  # If the node is gray, it means we have a cycle
            raise ValueError(f"Circular dependency detected: {schema_name} depends on {i} and vice versa")
        if visited[i] == 0:  # If the node is white, visit it
            topological_sort(i, visited, stack, dependencies)

    visited[schema_name] = 2  # Black
    stack.appendleft(schema_name)


class SchemaStructure(BaseModel):
    schemas: List[pa.DataFrameSchema] = Field(default_factory=list)
    document_schema: Optional[pa.DataFrameSchema] = None

    @validator('schemas', pre=True, each_item=True)
    def parse_schema(cls, v: Union[pa.DataFrameModel, pa.DataFrameSchema]):
        return v.to_schema() if hasattr(v, 'to_schema') else v

    @classmethod
    def from_dir(cls, dir_path: str, exclude: List[str]=[]):
        schemas = {}
        if os.path.isdir(dir_path):
            schema_paths = sorted(glob(os.path.join(dir_path, '*.json')), key=lambda x: not x.endswith('.json'))
        else:
            schema_paths = sorted(glob(dir_path), key=lambda x: not x.endswith('.json'))

        for filepath in schema_paths:
            try:
                if filepath.endswith('.json'):
                    schema = from_json(filepath)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    schema = from_yaml(filepath)
                else:
                    continue

                if schema.name in schemas or schema.name in exclude:
                    continue

                schemas[schema.name] = schema
            except Exception as e:
                logging.warning(f"Ignoring failed schema loading from '{filepath}': \n{e}")

        return cls(schemas=list(schemas.values()))

    @property
    def downstream_dependencies(self) -> Dict[str, List[str]]:
        dependencies = {}
        for schema in self.schemas:
            dependencies[schema.name] = [dep.name for dep in self.schemas \
                                         if f"{schema.name}_ref".lower() in dep.index.names]
        return dependencies

    @property
    def upstream_dependencies(self) -> Dict[str, List[str]]:
        dependencies = {}
        for schema in self.schemas:
            dependencies[schema.name] = [
                other.name for other in self.schemas \
                if f"{other.name}_ref".lower() in (schema.index.names or [schema.index.name])]

        return dependencies

    def index_names(self, schema: str) -> List[str]:
        return list(self.__getitem__(schema).index.names or [self.__getitem__(schema).index.name])

    def columns(self, schema: str) -> List[str]:
        columns = list(self.__getitem__(schema).columns)
        return columns

    @property
    def ordering(self) -> List[str]:
        visited = {schema.name: 0 for schema in self.schemas}
        stack = deque()

        for schema in self.schemas[::-1]:  # Reverse order to ensure the same order as the input
            if visited[schema.name] == 0:
                # If the node is white, visit it
                topological_sort(schema.name, visited, stack, self.downstream_dependencies)

        return list(stack)

    def __iter__(self):
        return iter(self.ordering)

    def __getitem__(self, item: str):
        if isinstance(item, pa.DataFrameSchema):
            item = item.name
        elif isinstance(item, MetaModel):
            item = str(item)

        for schema in self.schemas:
            if schema.name == item:
                return schema
        raise KeyError(f"No schema found for '{item}'")

    def __repr_args__(self):
        args = [(s.name, (s.index.names or [s.index.name]) + list(s.columns)) for s in self.schemas]
        return args

    class Config:
        arbitrary_types_allowed = True


class PaperExtraction(BaseModel):
    extractions: Dict[str, pd.DataFrame] = Field(default_factory=dict)
    schemas: SchemaStructure = Field(default_factory=SchemaStructure)
    durations: Dict[str, Optional[float]] = Field(default_factory=dict)
    reference: Optional[str] = None

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
