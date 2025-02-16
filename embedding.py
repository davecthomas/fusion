from abc import ABC, abstractmethod
from typing import Any, List, Dict

# file: embedding.py

from abc import ABC, abstractmethod
from typing import Any, List, Dict

###############################################################################
#                            ABSTRACT BASE CLASSES
###############################################################################


class EmbeddingBase(ABC):
    """
    Abstract base class for all embedding classes (text, numeric, category, etc.).
    """

    @abstractmethod
    def embed(self, value: Any) -> List[float]:
        """
        Legacy single-value embedding interface (still used by numeric/category).
        """
        pass


class BooleanEmbedder(EmbeddingBase):
    """
    Boolean embedding as a single dimension:
      - True  -> [1.0]
      - False -> [0.0]
    """

    def embed(self, value: Any) -> List[float]:
        return [1.0] if bool(value) else [0.0]


class TextEmbedderBase(EmbeddingBase):
    """
    Abstract base class for text embedding.

    Adds single & batch text embedding interfaces:
      - embed_single(text)
      - embed_batch(texts)
    """

    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """
        Return the text embedding for the given string (single input).
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Return a list of embeddings for a list of input strings.
        """
        pass


###############################################################################
#                            TEXT EMBEDDING CLASSES
###############################################################################
import time
from typing import List

from openai import OpenAI
from openai import RateLimitError, BadRequestError, OpenAIError
from openai.types import EmbeddingCreateParams, CreateEmbeddingResponse


class OpenAITextEmbedder:
    """
    Text embedding using the newer OpenAI.embeddings.create(...) API
    with batch support and a cascaded retry/backoff logic.
    """

    def __init__(
        self, embedding_model: str, api_key: str, backoff_delays: List[int] = None
    ):
        """
        :param embedding_model: e.g., "text-embedding-ada-002" (1536 dimensions)
        :param api_key: OpenAI API key
        :param backoff_delays: A list of retry delay intervals in seconds.
                               e.g. [1, 2, 4] means 1s, then 2s, then 4s if failures recur.
        """
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.backoff_delays = backoff_delays or [1, 2, 4]
        self.client = OpenAI(api_key=self.api_key)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of input strings and returns a list of embeddings
        (one embedding list per string). If using "text-embedding-ada-002",
        each embedding will be 1536 floats.

        Retries on RateLimitError, BadRequestError, and other OpenAIError using
        the specified backoff delays.
        """
        params: EmbeddingCreateParams = {
            "input": texts,  # List of input texts
            "model": self.embedding_model,  # e.g. "text-embedding-ada-002"
        }

        max_retries = len(self.backoff_delays)
        attempt = 0

        while attempt < max_retries:
            try:
                response: CreateEmbeddingResponse = self.client.embeddings.create(
                    **params
                )
                # Extract embeddings from the response
                embeddings = [item.embedding for item in response.data]
                return embeddings

            except RateLimitError as e:
                # Hit rate limit - wait and retry
                if attempt < max_retries - 1:
                    delay = self.backoff_delays[attempt]
                    print(f"RateLimitError: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    raise e

            except BadRequestError as e:
                # The request was invalid (400). Possibly fix the request if feasible.
                if attempt < max_retries - 1:
                    delay = self.backoff_delays[attempt]
                    print(f"BadRequestError: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    raise e

            except OpenAIError as e:
                # Catch-all for other OpenAI-related issues
                if attempt < max_retries - 1:
                    delay = self.backoff_delays[attempt]
                    print(f"OpenAIError: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    raise e

    def embed_single(self, text: str) -> List[float]:
        """
        Convenience method for a single text. Reuses embed_batch.
        Returns a single embedding (list of floats) or raises if empty.
        """
        results = self.embed_batch([text])
        if results:
            return results[0]
        else:
            # This theoretically shouldn't happen unless 'data' was empty
            raise ValueError("No embedding returned for single input.")


###############################################################################
#                            EMBEDDING FACTORY
###############################################################################


class EmbeddingFactory:
    """
    Factory for creating text embedders based on a config dict.
    Extend this to handle multiple providers (OpenAI, Titan, etc.).
    """

    @staticmethod
    def create_text_embedder(config: Dict[str, Any]) -> TextEmbedderBase:
        """
        Creates an appropriate TextEmbedderBase instance based on config.
        Expected keys in config:
          - provider: "openai" or "huggingface" or ...
          - model_name: model name for the chosen provider
          - api_key: if using OpenAI
        """
        provider = config.get("provider", "openai")

        if provider == "openai":
            # Default to "text-embedding-ada-002" if not specified
            model_name = config.get("model_name", "text-embedding-ada-002")
            api_key = config["api_key"]  # Must be provided
            return OpenAITextEmbedder(model_name, api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}")


###############################################################################
#                         NUMERIC & CATEGORY EMBEDDINGS
###############################################################################


class NumericEmbedder(EmbeddingBase):
    """
    Numeric embedding using min-max normalization (1D vector).
    """

    def __init__(self, min_val: float, max_val: float):
        """
        min_val: minimum value in the dataset/column
        max_val: maximum value in the dataset/column
        """
        self.min_val = min_val
        self.max_val = max_val

    def embed(self, value: float) -> List[float]:
        """
        Normalize the value into [0,1].
        If max_val == min_val, returns [0.0] to avoid divide-by-zero.
        """
        if self.max_val == self.min_val:
            return [0.0]
        normalized = (value - self.min_val) / (self.max_val - self.min_val)
        return [normalized]


class CategoryEmbedder(EmbeddingBase):
    """
    Categorical embedding using simple one-hot encoding.
    """

    def __init__(self, categories: List[str]):
        """
        categories is the list of possible categories in a known order.
        """
        self.categories = categories
        self.category_to_idx = {cat: i for i, cat in enumerate(categories)}

    def embed(self, category: str) -> List[float]:
        """
        Return a one-hot vector.
        If the category is not in the list, you could either handle as unknown or skip.
        """
        vector = [0.0] * len(self.categories)
        idx = self.category_to_idx.get(category)
        if idx is not None:
            vector[idx] = 1.0
        return vector
