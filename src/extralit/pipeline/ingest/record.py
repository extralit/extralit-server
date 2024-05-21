from datetime import datetime
from typing import Optional, Union, List, Dict, Any

import argilla as rg
from argilla.client.feedback.schemas.remote.records import RemoteFeedbackRecord
from argilla.client.sdk.users.models import UserModel

from extralit.convert.json_table import is_json_table
from extralit.pipeline.update.suggestion import get_record_suggestion_value


def get_record_data(record: Union[RemoteFeedbackRecord, rg.FeedbackRecord],
                    fields: Union[List[str], str],
                    answers: Optional[Union[List[str], str]] = None,
                    suggestions: Optional[Union[List[str], str]] = None,
                    metadatas: Optional[Union[List[str], str]] = None,
                    users: Optional[Union[List[rg.User], rg.User]] = None,
                    skip_status: Optional[List[str]] = ["discarded"]) \
        -> Dict[str, Any]:
    fields = [fields] if isinstance(fields, str) else set(fields) if fields else []
    answers = [answers] if isinstance(answers, str) else set(answers) if answers else []
    suggestions = [suggestions] if isinstance(suggestions, str) else set(suggestions) if suggestions else []
    metadatas = [metadatas] if isinstance(metadatas, str) else set(metadatas) if metadatas else []
    users = [users] if isinstance(users, (UserModel, rg.User)) else list(users) if users else []
    responses = record.responses
    if suggestions:
        assert users, "Users must be provided to get suggestions"

    if users:
        user_ids = [u.id for u in users]
        responses = [r for r in responses if r.user_id in user_ids]

    if skip_status and any(r.status in skip_status for r in responses):
        return {}

    data = {}
    for field in fields:
        if field in record.fields:
            data[field] = record.fields[field]

    selected_response = next((r for r in responses[::-1]), None)
    for answer in answers:
        if selected_response and answer in selected_response.values:
            data[answer] = selected_response.values[answer].value

    for suggestion in suggestions:
        data[suggestion] = get_record_suggestion_value(record, question_name=suggestion, users=users)

    for key in metadatas:
        if key in record.metadata and key not in data:
            data[key] = record.metadata[key]

    return data


@DeprecationWarning
def get_record_table(record: Union[RemoteFeedbackRecord, rg.FeedbackRecord],
                     field='extraction',
                     answer='extraction-correction',
                     users: Optional[Union[List[rg.User], rg.User]] = None,
                     skip_status=["discarded"]) -> Optional[str]:
    value = None
    outputs = get_record_data(record,
                              fields=field,
                              answers=answer,
                              users=users,
                              skip_status=skip_status)

    if answer in outputs and is_json_table(outputs[answer]):
        value = outputs[answer]
    elif field in outputs and is_json_table(outputs[field]):
        value = outputs[field]

    return value


def get_record_timestamp(record: Union[RemoteFeedbackRecord, rg.FeedbackRecord]) -> Optional[datetime]:
    timestamp = record.updated_at or record.inserted_at

    if len(record.responses):
        response = record.responses[-1]
        response_timestamp = response.updated_at or response.inserted_at
        if response_timestamp and response_timestamp > timestamp:
            timestamp = response_timestamp

    return timestamp
