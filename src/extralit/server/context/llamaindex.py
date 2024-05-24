import logging
import os
from typing import Optional

from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings, set_global_handler
_LOGGER = logging.getLogger(__name__)


def get_langfuse_callback(public_key: Optional[str] = None, secret_key: Optional[str] = None) -> LlamaIndexCallbackHandler:
    try:
        langfuse_callback_handler = LlamaIndexCallbackHandler(
            host=os.getenv('LANGFUSE_HOST'),
            public_key=public_key if public_key else os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=secret_key if secret_key else os.getenv('LANGFUSE_SECRET_KEY'),
        )
        if not Settings.callback_manager.handlers:
            Settings.callback_manager.add_handler(langfuse_callback_handler)
        set_global_handler("langfuse")
    except Exception as e:
        _LOGGER.error(f"Failed to create Langfuse callback handler: {e}")
        langfuse_callback_handler = None

    return langfuse_callback_handler
