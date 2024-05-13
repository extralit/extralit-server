import logging
from typing import List, Dict, Any, Optional, Union

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator, FilterCondition,
)
from llama_index.vector_stores.weaviate.base import _to_weaviate_filter
from llama_index.vector_stores.weaviate.utils import parse_get_response
from weaviate import Client


def query_weaviate_db(weaviate_client: Client,
                      index_name: str,
                      filters: Union[Dict[str, Any], MetadataFilters],
                      properties: Union[List, Dict] = ['reference', 'header', 'doc_id'],
                      limit: Optional[int] = None) -> List[Dict[str, Any]]:

    if not weaviate_client.is_ready():
        raise ValueError("Weaviate client is not ready")
    elif not weaviate_client.schema.contains({'class': index_name, 'properties': {}}):
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

    query_builder = weaviate_client.query \
        .get(index_name, properties) \
        .with_where(_to_weaviate_filter(filters))
    if limit is not None:
        query_builder.with_limit(limit)

    query_result = query_builder.do()
    parsed_result = parse_get_response(query_result)
    entries = parsed_result[index_name]
    return entries


def delete_from_weaviate_db(weaviate_client: Client, doc_ids: List[str], index_name: str) -> int:
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


def vectordb_has_document(paper, weaviate_client, index_name):
    has_document_in_vecstore = query_weaviate_db(
        weaviate_client, index_name, filters={'reference': paper.name}, properties=['doc_id', 'reference'], limit=1)
    return has_document_in_vecstore
