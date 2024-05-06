from typing import Union, List

import argilla as rg

__all__ = ['update_record_suggestions']

from argilla.client.feedback.schemas.remote.records import RemoteFeedbackRecord


def update_record_suggestions(
        record: RemoteFeedbackRecord,
        suggestions: Union[rg.SuggestionSchema, List[rg.SuggestionSchema]]) \
        -> rg.FeedbackRecord:
    if not isinstance(suggestions, list):
        suggestions = [suggestions]

    # Create a dictionary from the new suggestions
    new_suggestions_dict = {
        (s.question_name, s.type, s.agent): s \
        for s in suggestions \
        if s.question_name in record.question_name_to_id}

    if new_suggestions_dict:
        # Keep only the suggestions that are not in the new suggestions
        updated_suggestions = [
            s for s in record.suggestions
            if (s.question_name, s.type, s.agent) not in new_suggestions_dict
        ]

        record.suggestions = updated_suggestions + list(new_suggestions_dict.values())

    return record