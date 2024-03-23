from enum import Enum

import requests


class Endpoints(Enum):
    reranker = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    embedding = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"


class CallEndpoints():
    def _construct_payload(self,end_point,**kwargs):

        if end_point == "reranker":
            payload = {
                "model": kwargs["model"],
                "query": {
                    "text": kwargs["input"]
                },
                "passages": kwargs["passages"]
            }

            invoke_url = Endpoints.reranker.value
            return invoke_url,payload

    def _construct_headers(self,**kwargs):

        api_key = kwargs["api_key"]
        headers = {
        "Authorization": "Bearer " + api_key,
        "Accept": "application/json",
        }

        return headers

    def return_response(self,end_point,**kwargs):

        headers = self._construct_headers(**kwargs)
        invoke_url,payload = self._construct_payload(end_point,**kwargs)

        session = requests.Session()

        response = session.post(invoke_url, headers=headers, json=payload)

        response.raise_for_status()
        response_body = response.json()

        return response_body



if __name__=="__main__":
    print(Endpoints.reranker.value)