import os

from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings, set_global_handler, BaseCallbackHandler


def get_langfuse_global() -> BaseCallbackHandler:
    langfuse_callback_handler = LlamaIndexCallbackHandler(
        host=os.getenv('LANGFUSE_HOST'),
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
        secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    )
    if not Settings.callback_manager.handlers:
        Settings.callback_manager.add_handler(langfuse_callback_handler)
    set_global_handler("langfuse")

    from llama_index.core import global_handler
    return global_handler
