import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def call_llm(prompt: str, json_mode: bool = False) -> str:
    time.sleep(1)  # avoid rate limit
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file or export it in the terminal."
        )

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://localhost",
        },
        json={
            "model": "openrouter/auto",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    data = response.json()

    if "error" in data:
        raise Exception(f"OpenRouter error: {data['error']}")

    return data["choices"][0]["message"]["content"]

def embed_texts(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    global _model
    try:
        _model
    except NameError:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model.encode(texts, convert_to_numpy=True).tolist()