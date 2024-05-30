import os
from typing import Optional
import weaviate
from weaviate import WeaviateClient


def get_weaviate_client() -> Optional[WeaviateClient]:
    if 'WCS_HTTP_URL' in os.environ:
        api_keys = os.getenv('WCS_API_KEY', '')

        # client = weaviate.Client(
        #     url=os.getenv("WCS_HTTP_URL", None),
        #     auth_client_secret=weaviate.auth.AuthApiKey(api_keys.split(',')[0])
        # )

        weaviate_client = weaviate.connect_to_custom(
            http_host=os.getenv("WCS_HTTP_URL"),
            http_port=80,
            http_secure=False,
            grpc_host=os.getenv('WCS_GRPC_URL'),
            grpc_port=50051,
            grpc_secure=False,
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WCS_API_KEY")),
            headers={
                "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
            }
        )

        return weaviate_client

    return None