import datetime
from typing import Dict, List, Optional

from minio.datatypes import Object
from argilla_server.pydantic_v1 import BaseModel, Field


class MinioObject(BaseModel):
    bucket_name: str
    object_name: str
    last_modified: datetime
    etag: str
    size: int
    content_type: str
    metadata: Dict[str, str]
    version_id: Optional[str]

    @classmethod
    def from_minio_object(cls, minio_object: Object):
        return cls(
            bucket_name=minio_object.bucket_name,
            object_name=minio_object.object_name,
            last_modified=minio_object.last_modified,
            etag=minio_object.etag,
            size=minio_object.size,
            content_type=minio_object.content_type,
            metadata=minio_object.metadata,
            version_id=minio_object.version_id,
        )


class FileResponse(BaseModel):
    current_version: Object
    previous_versions: List[Object]
