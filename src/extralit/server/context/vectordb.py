import os
from typing import Optional
import weaviate

def get_weaviate_client() -> Optional[weaviate.Client]:
    if 'WCS_HTTP_URL' in os.environ:
        client = weaviate.Client(
            url=os.getenv("WCS_HTTP_URL"),
            auth_client_secret=weaviate.auth.AuthApiKey(os.getenv('WCS_API_KEY'))
        )
        return client

    return None