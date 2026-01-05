from dataclasses import dataclass
from openai import OpenAI
#from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass
class QueryParam:
    response_type: str = "Keep the responses as brief and accurate as possible. If you need to present information in a list format, use (1), (2), (3), etc., instead of numbered bullets like 1., 2., 3. "
    top_k: int = 10
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    # alpha: int = 0.5
    number_of_mmentities: int = 3

@dataclass
class EmbeddingParam:
    embedding_dim: int = 1024
    max_token_size: int = 4096

EMBED_CONFIG = EmbeddingParam()

cache_path = './cache'
mineru_dir = "./example_input/mineru_result"

# embedding_model_dir = './cache/all-MiniLM-L6-v2'
# EMBED_MODEL = SentenceTransformer(embedding_model_dir, device="cpu")
# EMBED_MODEL = SentenceTransformer(embedding_model_dir, trust_remote_code=True, device="cuda:0")

# def encode(content):
#     return EMBED_MODEL.encode(content)
# """
# def encode(content):
#     return EMBED_MODEL.encode(content, prompt_name="s2p_query", convert_to_tensor=True).cpu()
# """

    
# LLM model parameters
API_KEY = os.getenv("LLM_MODEL_API_KEY")
MODEL = os.getenv("LLM_MODEL_NAME")
URL = os.getenv("LLM_MODEL_BASE_URL")
# VLM model parameters
MM_API_KEY = os.getenv("VLM_MODEL_API_KEY")
MM_MODEL = os.getenv("VLM_MODEL_NAME")
MM_URL = os.getenv("VLM_MODEL_BASE_URL")
# Embedding model parameters
EMBEDDING_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_URL = os.getenv("EMBEDDING_MODEL_BASE_URL")

embed_client = OpenAI(
    api_key=EMBEDDING_API_KEY, 
    base_url=EMBEDDING_URL
)

def encode(texts):
    response = embed_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return np.array([item.embedding for item in response.data])