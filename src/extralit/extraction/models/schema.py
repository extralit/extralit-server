import logging
import os
from collections import deque
from glob import glob
from io import BytesIO
from typing import List, Optional, Union, Dict

import pandera as pa
from minio import Minio
from pandera.api.base.model import MetaModel
from pandera.io import from_json, from_yaml
from pydantic.v1 import BaseModel, Field, validator


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

    @classmethod
    def from_s3(cls, workspace: str, minio_client: Minio, prefix: str = 'schemas/',
                exclude: List[str] = []):
        schemas = {}
        objects = minio_client.list_objects(workspace, prefix=prefix, include_version=False)

        for obj in objects:
            filepath = obj.object_name
            try:
                data = minio_client.get_object(workspace, filepath)
                file_data = BytesIO(data.read())

                if filepath.endswith('.json'):
                    schema = from_json(file_data)
                elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    schema = from_yaml(file_data)
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

        for schema in self.schemas:
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
            if schema.name.lower() == item.lower():
                return schema
        raise KeyError(f"No schema found for '{item}'")

    def __repr_args__(self):
        args = [(s.name, (s.index.names or [s.index.name]) + list(s.columns)) for s in self.schemas]
        return args

    class Config:
        arbitrary_types_allowed = True


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
