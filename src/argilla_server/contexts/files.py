import logging
from typing import Optional
from urllib.parse import urlparse
from argilla_server.settings import settings
from fastapi import HTTPException
from minio import Minio, S3Error
from minio.versioningconfig import VersioningConfig
from minio.datatypes import Object

_LOGGER = logging.getLogger("argilla")


def get_minio_client() -> Optional[Minio]:
    if None in [settings.s3_endpoint, settings.s3_access_key, settings.s3_secret_key]:
        return None

    try:
        parsed_url = urlparse(settings.s3_endpoint)
        hostname = parsed_url.hostname
        port = parsed_url.port

        if hostname is None:
            print(f"Invalid URL: no hostname found, possible due to lacking http(s) protocol. Given '{settings.s3_endpoint}'")
            return None

        return Minio(
            endpoint=f'{hostname}:{port}' if port else hostname,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=parsed_url.scheme == "https",
        )
    except Exception as e:
        _LOGGER.error(f"Error creating Minio client: {e}")
        return None


async def stat_object(minio_client: Minio, bucket: str, object: str, version_id: Optional[str] = None):
    try:
        return minio_client.stat_object(bucket, object, version_id=version_id)
    except S3Error as se:
        _LOGGER.error(f"Error getting object {object} from bucket {bucket}: {se}")
        raise HTTPException(status_code=404, detail=f"Object {object} not found in bucket {bucket}")
    except Exception as e:
        _LOGGER.error(f"Error getting object {object} from bucket {bucket}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")



async def get_object(minio_client: Minio, bucket: str, object: str, version_id: Optional[str] = None):
    try:
        stat = minio_client.stat_object(bucket, object, version_id=version_id)

        # Get the object
        obj = minio_client.get_object(bucket, object, version_id=stat.version_id)

        # Get the current version
        current_version = await minio_client.get_object_version(bucket, object)

        # Get the list of previous versions
        previous_versions = await minio_client.list_object_versions(bucket, prefix=object)

        # Filter out the current version from the list of previous versions
        previous_versions = [version for version in previous_versions if version.version_id != current_version.version_id]

        # return ObjectData(current_version=current_version, previous_versions=previous_versions)
    
    except S3Error as se:
        _LOGGER.error(f"Error getting object {object} from bucket {bucket}: {se}")
        raise HTTPException(status_code=404, detail=f"Object {object} not found in bucket {bucket}")
    except Exception as e:
        _LOGGER.error(f"Error getting object {object} from bucket {bucket}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    


# async def put_object(minio_client: Minio, bucket: str, object: str, data: bytes):


async def create_bucket(minio_client: Minio, workspace_name: str):
    try:
        await minio_client.make_bucket(workspace_name)
        await minio_client.set_bucket_versioning(workspace_name, VersioningConfig(VersioningConfig.ENABLED))
    except S3Error as se:
        if se.code == "BucketAlreadyOwnedByYou":
            pass
        else:
            _LOGGER.error(f"Error creating bucket {workspace_name}: {se}")
            raise se
    except Exception as e:
        _LOGGER.error(f"Error creating bucket {workspace_name}: {e}")
        raise e


async def delete_bucket(minio_client: Minio, workspace_name: str):
    try:
        await minio_client.remove_bucket(workspace_name)
    except S3Error as se:
        if se.code == "NoSuchBucket":
            pass
        else:
            _LOGGER.error(f"Error creating bucket {workspace_name}: {se}")
            raise se
    except Exception as e:
        _LOGGER.error(f"Error deleting bucket {workspace_name}: {e}")
        raise e