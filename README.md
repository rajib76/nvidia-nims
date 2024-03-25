# NVIDIA SERVICES

NVIDIA recently announced NVIDIA NIMs which offers optimized inference
microservices for deploying AI models at scale. The NIM services
along with the NEMO services will allow to develop and deploy RAG based
applications quickly in production. 

This package is created to be the PYTHON SDK for those services. The 
idea is to write only few lines of code to develop applications with NVIDIA
services. The bolier plate code goes in the SDK

# How to use the SDK

The SDK is now pushed to PYPI. To install it run the below command

`pip install nvidia-services`

There are two services which are now part of the SDK

1. EMBEDDING
2. RERANKING

Example code for embedding

``` Python 
import os
from dotenv import load_dotenv
from nvidia_services.embeddings.nvidia_embeddings import NVIDIAEmbeddings
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nv = NVIDIAEmbeddings(api_key=NVIDIA_API_KEY)
embeddings = nv.create_embed(["hello, how are you","I am fine"])
for embedding in embeddings:
    print(embedding)
```
Example code for reranker

``` Python
import os
from dotenv import load_dotenv
from nvidia_services.retrievals.nvidia_reranker import NVDIARerankerMistral
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
nv = NVDIARerankerMistral(api_key=NVIDIA_API_KEY)
passages = [
    {
      "text": "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to 10X higher performance for applications running terabytes of data."
    },
    {
      "text": "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets."
    },
    {
      "text": "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™."
    }
  ]
input = "What is the GPU memory bandwidth of H100 SXM?"
result = nv.return_context(input=input,passages=passages)
print(result) 
```

Example code to call the mistral model

```Python
import os

from dotenv import load_dotenv

from nvidia_services.models.mistralai_models import MistralAIModels

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
mistral = MistralAIModels(api_key=NVIDIA_API_KEY)

prompt = "Where is TajMahal?"
result = mistral.generate_response(prompt=prompt)

for chunk in result:
    print(chunk.choices[0].delta.content)


```
