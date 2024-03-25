from abc import abstractmethod

from pydantic.v1 import BaseModel


class BaseLMModel(BaseModel):

    @abstractmethod
    def generate_response(self,prompt:str,**kwargs):
        pass

    @abstractmethod
    def agenerate_response(selfprompt:str,**kwargs):
        pass

