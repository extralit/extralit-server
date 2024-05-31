import logging
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, Depends
from starlette.requests import Request
from starlette.responses import StreamingResponse

from argilla_server.models import User
from argilla_server.security import auth
from argilla_server.settings import settings

_LOGGER = logging.getLogger("models")

router = APIRouter(tags=["models"])

@router.api_route("/models/{rest_of_path:path}", 
                  methods=["GET", "POST", "PUT", "DELETE"], 
                  response_class=StreamingResponse)
async def proxy(request: Request, rest_of_path: str,
                current_user: User = Depends(auth.get_optional_current_user)):
    url = urljoin(settings.extralit_url, rest_of_path)
    params = dict(request.query_params)
    if current_user:
        params['username'] = current_user.username

    _LOGGER.info(f'PROXY {url} {params}')

    client = httpx.AsyncClient(timeout=60.0)
    if request.method == "GET":
        request = client.build_request("GET", url, params=params)
    elif request.method == "POST":
        data = await request.json()
        request = client.build_request("POST", url, json=data, params=params)
    elif request.method == "PUT":
        data = await request.json()
        request = client.build_request("PUT", url, data=data, params=params)
    elif request.method == "DELETE":
        request = client.build_request("DELETE", url, params=params)
    else:
        return {"message": "Method not supported"}

    async def stream_response():
        response = await client.send(request, stream=True)
        async for chunk in response.aiter_raw():
            yield chunk
        client.aclose()

    return StreamingResponse(stream_response(), media_type="text/event-stream")
    