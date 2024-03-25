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