# Author: Rajib
# TO do :
# Add validation, retry with tenacity, async implementation

import requests
from pydantic.v1 import validator, Field

from nvidia_services.embeddings.base import BaseEmbeddings


class NVIDIAEmbeddings(BaseEmbeddings):
    invoke_url: str = Field(default="https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings")
    api_key: str
    _embeddings = []

    @validator('api_key')
    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        if api_key is None:
            raise ValueError('please input the api_key')
        return api_key

    def create_embed(self, text: []):
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Accept": "application/json",
        }

        payload = {
            "input": text,
            "input_type": "query",
            "model": "NV-Embed-QA"
        }

        # re-use connections
        session = requests.Session()

        response = session.post(self.invoke_url, headers=headers, json=payload)

        response.raise_for_status()
        response_body = response.json()
        embeddings = response_body['data']
        for embedding in embeddings:
            self._embeddings.append(embedding['embedding'])
        return self._embeddings

    async def acreate_embed(self, text: []):
        pass
