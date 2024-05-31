import logging
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, Depends
from starlette.requests import Request
from starlette.responses import StreamingResponse

from argilla_server.models import User
from argilla_server.security import auth
from argilla_server.settings import settings

_LOGGER = logging.getLogger("files")

router = APIRouter(tags=["models"])

@router.api_route("/models/{rest_of_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, rest_of_path: str,
                current_user: User = Depends(auth.get_optional_current_user)):
    url = urljoin(settings.extralit_url, rest_of_path)
    params = dict(request.query_params)
    if current_user:
        params['username'] = current_user.username

    _LOGGER.info(f'PROXY {url} {params}')

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.request(
            method=request.method, url=url, params=params, 
            data=await request.body(), headers=request.headers, 
            stream=True, )
        # if request.method == "GET":
        #     r = await client.get(url, params=params)
        # elif request.method == "POST":
        #     data = await request.json()
        #     r = await client.post(url, json=data, params=params)
        # elif request.method == "PUT":
        #     data = await request.json()
        #     r = await client.put(url, data=data, params=params)
        # elif request.method == "DELETE":
        #     r = await client.delete(url, params=params)
        # else:
        #     return {"message": "Method not supported"}

    async def content_generator():
        async for chunk in r.aiter_bytes():
            yield chunk

    return StreamingResponse(content_generator(), status_code=r.status_code, 
                             headers=r.headers, media_type=r.headers.get('content-type'))

