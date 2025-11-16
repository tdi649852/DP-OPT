"""OpenRouter LLM access for testing multiple models."""
import os
import time
import requests
from tqdm import tqdm
from typing import List, Union, Dict, Optional

RETRY_TIMEOUT = 20
MAX_RETRIES = 3


class OpenRouterLLM:
    """Wrapper for OpenRouter API to access multiple LLM models."""

    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 temperature: float = 0.9,
                 max_tokens: int = 128,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 disable_tqdm: bool = False):
        """Initialize the OpenRouter LLM.

        Args:
            model: The model to use (e.g., "anthropic/claude-3-opus", "meta-llama/llama-3-8b-instruct")
            api_key: OpenRouter API key (if not provided, will use OPENROUTER_API_KEY env var)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            disable_tqdm: Disable progress bar
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided either as argument or "
                "through OPENROUTER_API_KEY environment variable"
            )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.disable_tqdm = disable_tqdm

        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/DP-OPT"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "DP-OPT")
        }

    def _make_request(self, messages: List[Dict[str, str]], n: int = 1) -> Dict:
        """Make a request to OpenRouter API with retry logic."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "n": n,
        }

        response = None
        retries = 0

        while response is None and retries < MAX_RETRIES:
            try:
                resp = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                resp.raise_for_status()
                response = resp.json()

                # Check for errors in response
                if "error" in response:
                    raise Exception(f"OpenRouter API error: {response['error']}")

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    raise Exception(f"Failed after {MAX_RETRIES} retries: {str(e)}")
                print(f"Request failed: {e}")
                print(f'Retrying... ({retries}/{MAX_RETRIES})')
                time.sleep(RETRY_TIMEOUT * retries)
            except Exception as e:
                retries += 1
                if retries >= MAX_RETRIES:
                    raise Exception(f"Failed after {MAX_RETRIES} retries: {str(e)}")
                print(f"Error: {e}")
                print(f'Retrying... ({retries}/{MAX_RETRIES})')
                time.sleep(RETRY_TIMEOUT * retries)

        return response

    def generate_text(self, prompts: Union[str, List[str]], n: int = 1) -> List[str]:
        """Generate text from prompts.

        Args:
            prompts: Single prompt string or list of prompt strings
            n: Number of completions to generate per prompt

        Returns:
            List of generated text strings
        """
        if not isinstance(prompts, list):
            prompts = [prompts]

        all_results = []

        # Process prompts with progress bar
        for prompt in tqdm(prompts, disable=self.disable_tqdm, desc="Generating with OpenRouter"):
            # Clean up the prompt
            clean_prompt = prompt.replace('[APE]', '').strip()

            # Prepare messages for chat format
            messages = [{"role": "user", "content": clean_prompt}]

            # Make API request
            response = self._make_request(messages, n=n)

            # Extract generated text from response
            if "choices" in response:
                for choice in response["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        all_results.append(choice["message"]["content"])
                    else:
                        # Fallback for different response formats
                        all_results.append(choice.get("text", ""))
            else:
                raise Exception(f"Unexpected response format: {response}")

        return all_results

    def batch_generate(self, prompts: List[str], batch_size: int = 1) -> List[str]:
        """Generate text in batches (OpenRouter processes one at a time).

        Args:
            prompts: List of prompt strings
            batch_size: Batch size (mainly for compatibility, processes sequentially)

        Returns:
            List of generated text strings
        """
        # OpenRouter API processes requests one at a time
        # This method exists for API compatibility
        return self.generate_text(prompts, n=1)

    def __call__(self, prompts: Union[str, List[str]], n: int = 1) -> List[str]:
        """Callable interface for generating text."""
        return self.generate_text(prompts, n=n)


def create_openrouter_llm(
    model: str = "meta-llama/llama-3-8b-instruct",
    temperature: float = 0.9,
    max_tokens: int = 128,
    repetition_penalty: float = 1.0,
    **kwargs
) -> OpenRouterLLM:
    """Factory function to create an OpenRouterLLM instance.

    Args:
        model: OpenRouter model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        repetition_penalty: Repetition penalty
        **kwargs: Additional arguments

    Returns:
        OpenRouterLLM instance
    """
    return OpenRouterLLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        **kwargs
    )
