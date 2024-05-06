import logging
import os.path
from collections import Counter
from os.path import join, exists
from typing import List, Optional

import argilla as rg
import pandas as pd
from llama_index.core import VectorStoreIndex, load_index_from_storage, global_handler
from llama_index.core.node_parser import SentenceSplitter, JSONNodeParser
from llama_index.core.schema import Document
from llama_index.core.service_context import ServiceContext
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbeddingMode, OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from weaviate import Client

from extralit.extraction.chunking import create_documents
from extralit.extraction.query import query_weaviate_db, delete_from_weaviate_db
from extralit.extraction.storage import get_storage_context

DEFAULT_RETRIEVAL_MODE = OpenAIEmbeddingMode.TEXT_SEARCH_MODE


def create_index(text_documents: List[Document], table_documents: List[Document],
                 weaviate_client: Optional[Client] = None, index_name: Optional[str] = "LlamaIndexDocumentSections",
                 persist_dir: Optional[str] = None,
                 embed_model='text-embedding-3-small', dimensions=1536, retrieval_mode=DEFAULT_RETRIEVAL_MODE,
                 chunk_size=4096, chunk_overlap=200, verbose=True, ) \
        -> VectorStoreIndex:
    logging.info(
        f"Creating index with {len(text_documents)} text and {len(table_documents)} table segments, `persist_dir={persist_dir}`")

    storage_context = get_storage_context(
        weaviate_client=weaviate_client,
        index_name=index_name,
        persist_dir=persist_dir)

    embedding_model = OpenAIEmbedding(
        mode=retrieval_mode, model=embed_model, dimensions=dimensions,
    )

    embed_model_context = ServiceContext.from_defaults(
        embed_model=embedding_model,
        node_parser=SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )
    index = VectorStoreIndex.from_documents(
        text_documents, storage_context=storage_context, service_context=embed_model_context)

    index.insert_nodes(
        table_documents, node_parser=JSONNodeParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    if persist_dir and not storage_context.vector_store:
        assert os.path.exists(persist_dir)
        index.storage_context.persist(persist_dir)

    if verbose:
        nodes_counts = Counter([doc.metadata['header'] for doc in index.docstore.docs.values()])
        nodes_counts = [(header, count) for header, count in nodes_counts.most_common() if count > 1]
        print(pd.DataFrame(nodes_counts, columns=['header', 'n_chunks'])) if nodes_counts else None

    return index


def create_or_load_vectorstore_index(paper: pd.Series,
                                     llm_model="gpt-3.5-turbo",
                                     embed_model='text-embedding-3-small',
                                     preprocessing_path='data/preprocessing/nougat/',
                                     preprocessing_dataset: rg.FeedbackDataset = None,
                                     reindex=False,
                                     weaviate_client: Optional[Client] = None,
                                     index_name: Optional[str] = "LlamaIndexDocumentSections",
                                     persist_dir='data/interim/vectorstore/',
                                     **kwargs) \
        -> VectorStoreIndex:
    """
    Creates or loads a VectorStoreIndex for a given paper.

    This function will either create a new VectorStoreIndex by processing the given paper, or load an existing one from
    the specified directory. If the `reindex` parameter is set to True, the function will reindex the paper even if an
    existing index is found.

    Args:
        paper (pd.Series): The paper to be indexed.
        llm_model (str, optional): The model to use for extraction. Defaults to 'gpt-3.5-turbo'.
        embed_model (str, optional): The model to use for embedding documents. Defaults to 'text-embedding-3-small'.
        preprocessing_path (str, optional): The path to the preprocessed data. Defaults to 'data/preprocessing/nougat/'.
        preprocessing_dataset (rg.FeedbackDataset, optional): Manually annotated preprocessing dataset. Defaults to None.
        reindex (bool, optional): If True, the index will be reindexed. Defaults to False.
        weaviate_client (Client, optional): The Weaviate client to use. Defaults to None.
        persist_dir (str, optional): The directory where the index is persisted. Defaults to 'data/interim/vectorstore/'.

    Returns:
        VectorStoreIndex: The created or loaded VectorStoreIndex.
    """
    local_dir = join(persist_dir, paper.name, embed_model)

    use_weaviate = weaviate_client is not None
    has_document_in_vecstore = query_weaviate_db(
        weaviate_client, index_name, filters={'reference': paper.name}, properties=['doc_id', 'reference'], limit=1)
    if reindex or (not exists(local_dir) and not has_document_in_vecstore):
        assert preprocessing_path is not None or preprocessing_dataset is not None, \
            "Either preprocessing_path or preprocessing_dataset must be given"
        text_documents, table_documents = create_documents(
            paper, preprocessing_path=preprocessing_path,
            preprocessing_dataset=preprocessing_dataset)

        if global_handler and hasattr(global_handler, 'set_trace_params'):
            global_handler.set_trace_params(
                name=f"embed-{paper.name}", tags=[paper.name]
            )

        if use_weaviate and has_document_in_vecstore:
            docs = query_weaviate_db(
                weaviate_client, index_name=index_name, filters={'reference': paper.name},
                properties=['doc_id', 'reference'],
                limit=None)
            delete_from_weaviate_db(weaviate_client, doc_ids=[doc['doc_id'] for doc in docs], index_name=index_name)

        create_index(
            text_documents, table_documents, weaviate_client=weaviate_client, index_name=index_name,
            persist_dir=local_dir, embed_model=embed_model)

    # Load the existing index
    storage_context = get_storage_context(weaviate_client=weaviate_client,
                                          index_name=index_name,
                                          persist_dir=local_dir)
    llm = OpenAI(model=llm_model, temperature=0.0, max_retries=3)
    service_context = ServiceContext.from_defaults(llm=llm)

    if not isinstance(storage_context.vector_store, SimpleVectorStore):
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store, service_context=service_context)
    else:
        index = load_index_from_storage(storage_context, service_context=service_context)

    return index


def load_index_retriever(paper: pd.Series, similarity_top_k=3,
                         embed_model='text-embedding-3-small',
                         vectorstore_path='data/interim/vectorstore/',
                         retrieval_mode=DEFAULT_RETRIEVAL_MODE,
                         **kwargs):
    persist_dir = join(vectorstore_path, paper.name, embed_model)
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

    llm = OpenAIEmbedding(model=embed_model,
                          mode=retrieval_mode)
    service_context = ServiceContext.from_defaults(
        embed_model=llm)

    index = load_index_from_storage(storage_context, service_context=service_context)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k, **kwargs)

    return retriever