# Author: Rajib Deb
# This abstracts the reranker NIMS of NVIDIA
# The reranker currently retruns the logits of the reranked documents
# This converts the logit into sigmoid values
import math
from typing import List

from pydantic.v1 import Field, validator

from nvidia_services.endpoints.nvidia_endpoints import Endpoints, CallEndpoints
from nvidia_services.retrievals.base import BaseRetrieval


class NVDIARerankerMistral(BaseRetrieval):
    # invoke_url: str = Field(default=Endpoints.reranker.value)
    reranker_model: str = Field(default="nv-rerank-qa-mistral-4b:1")
    api_key: str

    @validator('api_key')
    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        if api_key is None:
            raise ValueError('please input the api_key')
        return api_key

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def return_context(self, input: str, **kwargs):
        passages = kwargs["passages"]
        try:
            if not isinstance(passages, List):
                raise ValueError()
        except ValueError:
            print("must pass a list of passages as a dictionary. [{'text':'sample passage']")
            exit(1)

        invoke_client = CallEndpoints()  # TO DO need to move to initialize
        response_body = invoke_client.return_response("reranker",
                                                      api_key=self.api_key,
                                                      model=self.reranker_model,
                                                      input=input,
                                                      passages=passages)

        rankings = response_body['rankings']
        reranked_documents = []
        reranked_document = {}
        for ranking in rankings:
            probability_score = self.sigmoid(ranking['logit'])
            reranked_document = {"documents": passages[ranking['index']], "probability": probability_score}
            reranked_documents.append(reranked_document)
            reranked_document = {}

        return reranked_documents

    async def areturn_context(self, input: str, **kwargs):
        pass
