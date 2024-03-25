import os

from dotenv import load_dotenv

from nvidia_services.embeddings.nvidia_embeddings import NVIDIAEmbeddings

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

nv = NVIDIAEmbeddings(api_key=NVIDIA_API_KEY)

embeddings = nv.create_embed(["hello, how are you","I am fine"])

for embedding in embeddings:
    print(embedding)