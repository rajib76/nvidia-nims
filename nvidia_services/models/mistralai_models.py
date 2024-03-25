from typing import Dict

from pydantic.v1 import Field, validator, root_validator

from nvidia_services.endpoints.nvidia_endpoints import CallEndpoints, Endpoints
from nvidia_services.models.base import BaseLMModel


class MistralAIModels(BaseLMModel):
    generation_model: str = Field(default="mistralai/mixtral-8x7b-instruct-v0.1")
    api_key: str
    temperature: float = 0.5
    top_p: int = 1
    max_tokens: int = 1024
    stream: bool = True

    @validator('api_key')
    @classmethod
    def validate_api_key(cls, api_key: str) -> str:
        if api_key is None:
            raise ValueError('please input the api_key')
        return api_key


    @root_validator()
    def validate_kwargs(cls, values: Dict) -> Dict:
        """Validate that return messages is not True."""
        try:
            if values.get('temperature', False):
                print("temperature not specified. Will use default of 0.5")
            if values.get('top_p', False):
                print("top_p not specified. Will use default of 1")
            if values.get('max_tokens', False):
                print("max_tokens not specified. Will use default of 1024")
            if values.get('stream', False):
                print("stream not specified. Will use default of True")
            return values
        except Exception as e:
            print(e)

    def generate_response(self, **kwargs):
        invoke_client = CallEndpoints()  # TO DO need to move to initialize
        model = self.generation_model
        prompt = kwargs["prompt"]
        response_body = invoke_client.return_genai_response(api_key=self.api_key,
                                                            model=model,
                                                            prompt=prompt,
                                                            temperature=self.temperature,
                                                            top_p=self.top_p,
                                                            max_tokens=self.max_tokens,
                                                            stream=self.stream
                                                            )

        return response_body

    def agenerate_response(self, prompt: str, **kwargs):
        pass
