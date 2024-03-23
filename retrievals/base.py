from abc import abstractmethod
from typing import Any, Optional

from pydantic.v1 import BaseModel


class BaseRetrieval(BaseModel):

    @abstractmethod
    def return_context(self, input: str, **kwargs):
        pass

    @abstractmethod
    async def areturn_context(self, input: str, **kwargs):
        pass
