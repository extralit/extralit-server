import logging
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator, FilterCondition,
)
from llama_index.vector_stores.weaviate.base import _to_weaviate_filter
from llama_index.vector_stores.weaviate.utils import parse_get_response, validate_client, class_schema_exists
from weaviate import Client, WeaviateClient


def get_nodes_metadata(weaviate_client: WeaviateClient,
                       filters: Union[Dict[str, Any], MetadataFilters],
                       index_name: str='LlamaIndexDocumentSections',
                       properties: Union[List, Dict] = ['header', 'page_number', 'type', 'reference', 'doc_id'],
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:

    validate_client(weaviate_client)
    if not class_schema_exists(weaviate_client, index_name):
        return []

    if isinstance(filters, dict):
        assert set(filters.keys()).issubset(
            properties), f"Filters {list(filters)} must be a subset of properties {list(properties)}"
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key=k,
                    value=v,
                    operator=FilterOperator.IN if isinstance(v, list) else FilterOperator.EQ)
                for k, v in filters.items()],
            condition=FilterCondition.AND
        )

    collection = weaviate_client.collections.get(index_name)

    query_result = collection.query.fetch_objects(
        filters=_to_weaviate_filter(filters),
        return_properties=properties,
        limit=limit,
    )

    entries = [o.properties for o in query_result.objects]
    return entries


def delete_from_weaviate_db(weaviate_client: WeaviateClient, doc_ids: List[str], index_name: str) -> int:
    """
    Delete documents from Weaviate database using their document IDs.

    Args:
        weaviate_client (Client): The Weaviate client.
        doc_ids (List[str]): The list of document IDs to delete.

    """
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    where_filter = {
        "path": ["doc_id"],
        "operator": "ContainsAny",
        "valueText": doc_ids,
    }

    query = (
        weaviate_client.query.get(index_name, ['doc_id'])
        .with_additional(["id"])
        .with_where(where_filter)
        .with_limit(10000)  # 10,000 is the max Weaviate can fetch
    )

    query_result = query.do()
    parsed_result = parse_get_response(query_result)
    entries = parsed_result[index_name]
    for entry in entries:
        weaviate_client.data_object.delete(entry["_additional"]["id"], index_name)

    logging.info(f"Deleted {len(entries)} documents from Weaviate index {index_name}")
    print(f"Deleted {len(entries)} documents from Weaviate index {index_name}")
    return len(entries)


def vectordb_contains_any(reference: str, weaviate_client: WeaviateClient, index_name: str) -> bool:
    if weaviate_client is None:
        return False

    has_document_in_vecstore = get_nodes_metadata(
        weaviate_client, index_name=index_name,
        filters={'reference': reference},
        properties=['doc_id', 'reference'],
        limit=1)

    return len(has_document_in_vecstore) > 0

