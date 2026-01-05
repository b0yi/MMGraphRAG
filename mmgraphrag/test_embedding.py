import requests
import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()


class RemoteEmbeddingWrapper:
    def __init__(self, api_url):
        self.api_url = api_url

    def get_embedding(self, text):
        response = requests.post(self.api_url, json={"input": text, "model": "stella"})
        response.raise_for_status()
        #print(response.json())
        embedding = response.json()['data'][0]['embedding']
        return np.array(embedding)

embed_model = RemoteEmbeddingWrapper(os.getenv("EMBEDDING_MODEL_BASE_URL"))
text = "Hello, world!"
embedding = embed_model.get_embedding(text)
print(embedding)