from datetime import datetime
from typing import Any, Dict, List, Optional

from minio.datatypes import Object
from argilla_server.pydantic_v1 import BaseModel, Field, validator
from urllib3 import HTTPResponse
from minio.helpers import ObjectWriteResult

class ObjectMetadata(BaseModel):
    bucket_name: str
    object_name: str
    last_modified: Optional[datetime]
    is_latest: Optional[bool]
    etag: Optional[str]
    size: Optional[int]
    content_type: Optional[str]
    version_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

    @classmethod
    def from_minio_object(cls, minio_object: Object):
        print(minio_object.metadata)
        return cls(
            bucket_name=minio_object.bucket_name,
            object_name=minio_object.object_name,
            last_modified=minio_object.last_modified,
            is_latest=minio_object.is_latest != 'false',
            etag=minio_object.etag,
            size=minio_object.size,
            content_type=minio_object.content_type,
            version_id=minio_object.version_id,
        )
    
    @classmethod
    def from_minio_write_response(cls, write_result: ObjectWriteResult):
        return cls(
            bucket_name=write_result.bucket_name,
            object_name=write_result.object_name,
            last_modified=write_result.last_modified,
            is_latest=None,
            etag=write_result.etag,
            size=None,
            content_type=write_result.http_headers.get('Content-Type'),
            version_id=write_result.version_id,
        )

class FileObject(BaseModel):
    response: HTTPResponse
    metadata: Optional[ObjectMetadata]
    versions: Optional[List[ObjectMetadata]]

    class Config:
        arbitrary_types_allowed = True

    @validator('metadata', 'versions', pre=True, each_item=True)
    def convert_minio_object(cls, v):
        if isinstance(v, Object):
            return ObjectMetadata.from_minio_object(v)
        return v


class ListObjectsResponse(BaseModel):
    objects: List[ObjectMetadata]

    @validator('objects', pre=True, each_item=True)
    def convert_objects(cls, v):
        if isinstance(v, Object):
            return ObjectMetadata.from_minio_object(v)
        return v

