"""
Core module for Ollama API Client.
Designed around real-time VRAM release (Zero-retention via keep_alive=0).
"""
import time
import requests
from typing import Dict, Any, Optional, List

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

class OllamaManager:
    """API wrapper class for Ollama."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def _post_with_retry(self, url: str, payload: dict, retries: int = 3, backoff: float = 3.0,
                         retry_on_500: bool = False) -> requests.Response:
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                return response
            except requests.HTTPError:
                # 500 = bad input or model crash — only retry if caller opts in
                if response.status_code == 500 and retry_on_500 and attempt < retries - 1:
                    wait = backoff * (attempt + 1)
                    print(f"[Ollama] 500 error, retry {attempt+1}/{retries-1} after {wait:.0f}s...")
                    time.sleep(wait)
                    last_exc = RuntimeError(f"HTTP 500 from {url}")
                    continue
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                # Transient network / cold-start issue — always retry
                last_exc = exc
                if attempt < retries - 1:
                    wait = backoff * (attempt + 1)
                    print(f"[Ollama] Connection error ({exc.__class__.__name__}), retry {attempt+1}/{retries-1} after {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                raise
        raise last_exc

    def generate(self, model: str, prompt: str, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None, keep_alive: int = 0) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive
        }
        if system: payload["system"] = system
        if options: payload["options"] = options

        # retry_on_500=True for generation — model may need warm-up time
        return self._post_with_retry(url, payload, retry_on_500=True).json().get("response", "")

    def chat(self, model: str, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None, keep_alive: int = 0) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive
        }
        if options: payload["options"] = options

        return self._post_with_retry(url, payload).json().get("message", {}).get("content", "")

    def get_embeddings(self, model: str, prompt: str, keep_alive: int = 0) -> List[float]:
        # Try new endpoint first (/api/embed), fall back to legacy (/api/embeddings)
        # retry_on_500=False — 500 on embeddings = bad input, skip immediately
        for url, payload, key in [
            (f"{self.base_url}/api/embed",
             {"model": model, "input": prompt, "keep_alive": keep_alive},
             None),  # key=None means parse embeddings[]
            (f"{self.base_url}/api/embeddings",
             {"model": model, "prompt": prompt, "keep_alive": keep_alive},
             "embedding"),
        ]:
            try:
                resp = self._post_with_retry(url, payload, retry_on_500=False)
                data = resp.json()
                if key is None:
                    embs = data.get("embeddings")
                    if embs and isinstance(embs[0], list):
                        return embs[0]
                else:
                    result = data.get(key, [])
                    if result:
                        return result
            except Exception:
                continue
        raise RuntimeError(f"Both embedding endpoints failed for model={model}")

    def unload_model(self, model: str):
        """Force Ollama to release 100% of model VRAM."""
        print(f"[VRAM Manager] Unloading model '{model}' from VRAM...")
        try:
            self.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass
