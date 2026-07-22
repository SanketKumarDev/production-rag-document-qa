import requests

from app.core.exceptions import LLMUnavailableError


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def generate(self, prompt: str) -> str:
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300,
                    },
                },
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()
            answer = data.get("response", "").strip()

            if not answer:
                raise LLMUnavailableError(
                    "Ollama returned an empty response."
                )

            return answer

        except requests.RequestException as exc:
            raise LLMUnavailableError(
                f"Could not reach Ollama at {self.base_url}"
            ) from exc