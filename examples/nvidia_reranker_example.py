import os

from dotenv import load_dotenv

from retrievals.nvidia_reranker import NVDIARerankerMistral

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