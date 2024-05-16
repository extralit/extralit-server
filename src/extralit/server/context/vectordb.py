import os
from typing import Optional
import weaviate

def get_weaviate_client() -> Optional[weaviate.Client]:
    if 'WCS_HTTP_URL' in os.environ:
        api_keys = os.getenv('WCS_API_KEY', '')

        client = weaviate.Client(
            url=os.getenv("WCS_HTTP_URL", None),
            auth_client_secret=weaviate.auth.AuthApiKey(api_keys.split(',')[0])
        )
        return client

    return None