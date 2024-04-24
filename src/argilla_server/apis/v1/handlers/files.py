# from fastapi import APIRouter, FastAPI, Depends, HTTPException, status, Security
# from fastapi.responses import StreamingResponse
# from minio import Minio

# from argilla_server.security import auth
# from argilla_server.models import User
# from argilla_server.contexts import accounts, datasets

# router = APIRouter(tags=["files"])

# def get_minio_client():
#     return Minio(
#         "minio:9000",
#         access_key="YOUR-ACCESS-KEY",
#         secret_key="YOUR-SECRET-KEY",
#         secure=True
#     )

# @router.get("/files/{bucket}/{object}")
# async def get_file(
#     bucket: str, object: str, 
#     minio_client: Minio = Depends(get_minio_client),
#     current_user: User = Security(auth.get_current_user)):

#     stat = await minio_client.stat_object(bucket, object)
#     workspace_id = stat.metadata.get("workspace_id")

#         # Check if the current user is in the workspace
#     workspace: Workspace = await workspaces.get_workspace_by_id(db, workspace_id)
#     if workspace is None or current_user.id not in [user.id for user in workspace.users]:
#         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have permission to access this file")


#     try:
#         response = await minio_client.get_object(bucket, object)
#         return StreamingResponse(response, media_type="application/pdf")
#     except Exception as e:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))