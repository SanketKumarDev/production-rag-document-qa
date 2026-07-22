import requests
from app.core.exceptions import LLMUnavailableError


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.RequestException as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}"
            ) from exc
