from typing import Optional
from argilla_server.policies import FilePolicy, _exists_workspace_user_by_user_and_workspace_name, authorize
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from minio import Minio, S3Error

from argilla_server.security import auth
from argilla_server.models import User
from argilla_server.contexts import files

router = APIRouter(tags=["files"])

@router.get("/files/{bucket}/{object}")
async def get_file(
    *,
    bucket: str, 
    object: str, 
    version_id: Optional[str] = None,
    minio_client: Minio = Depends(files.get_minio_client),
    current_user: User = Security(auth.get_current_user)):

    # Check if the current user is in the workspace to have access to the s3 bucket of the same name
    await authorize(current_user, FilePolicy.get(bucket))

    stat = minio_client.stat_object(bucket, object, version_id=version_id)

    try:
        response = minio_client.get_object(bucket, object, version_id=stat.version_id)
        headers = {
            "Content-Disposition": f"attachment; filename={object}",
            "Content-Type": stat.content_type,
            "X-Version-Id": stat.version_id,
            "X-Last-Modified": stat.last_modified.strftime('%Y-%m-%dT%H:%M:%S'),
            "X-Is-Latest": stat.is_latest != 'false',
        }
        return StreamingResponse(response, media_type=stat.content_type, headers=headers)
    
    except S3Error as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    
