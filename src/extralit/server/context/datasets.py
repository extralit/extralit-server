import os
from typing import Optional

import argilla as rg
from argilla.client.feedback.dataset.remote.dataset import RemoteFeedbackDataset
from argilla.client.sdk.commons.errors import UnauthorizedApiError


def get_argilla_dataset(name="Table-Preprocessing", workspace="itn-recalibration") -> RemoteFeedbackDataset:
    try:
        rg.init(
            api_url=os.getenv('ARGILLA_BASE_URL'),
            api_key=os.getenv('ARGILLA_API_KEY'),
            workspace='argilla',
        )
    except Exception as e:
        print(e)

    dataset = rg.FeedbackDataset.from_argilla(name=name, workspace=workspace, with_documents=False)

    return dataset
