import os
from typing import Optional

import argilla as rg

def get_argilla_client() -> Optional:
    print("Initializing Argilla client...", os.getenv('ARGILLA_BASE_URL'), os.getenv('ARGILLA_API_KEY'))

    try:
        client = rg.init(
            api_url=os.getenv('ARGILLA_BASE_URL'),
            api_key=os.getenv('ARGILLA_API_KEY'),
        )
        return client
    except Exception as e:
        print(f"Error initializing Argilla client: {e}")

    return None
