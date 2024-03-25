from abc import abstractmethod
from typing import Any

from pydantic.v1 import BaseModel

class BaseEmbeddings(BaseModel):

    @abstractmethod
    def create_embed(self,text:[]):
        pass


    @abstractmethod
    async def acreate_embed(self,text:[]):
        pass