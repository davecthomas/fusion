from abc import ABC, abstractmethod
from typing import Any, List, Dict
from openai import openai

###############################################################################
#                            ABSTRACT BASE CLASSES
###############################################################################


class EmbeddingBase(ABC):
    """
    Abstract base class for all embedding classes.
    """

    @abstractmethod
    def embed(self, value: Any) -> List[float]:
        """
        Take a raw value (text, numeric, category, etc.) and return a numeric vector.
        """
        pass


class TextEmbedderBase(EmbeddingBase):
    """
    Abstract base class for text embedding.
    """

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Return the text embedding for the given text.
        """
        pass


###############################################################################
#                            TEXT EMBEDDING CLASSES
###############################################################################


class OpenAITextEmbedder(TextEmbedderBase):
    """
    Text embedding using the OpenAI Embeddings API.
    """

    def __init__(self, model_name: str, api_key: str):
        """
        model_name: Name of the OpenAI embedding model, e.g., "text-embedding-ada-002"
        api_key: The OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key

        # Set the global OpenAI API key (you could also store it locally if you prefer)
        openai.api_key = self.api_key

    def embed(self, text: str) -> List[float]:
        """
        Uses OpenAI's Embedding API to generate embeddings for the given text.
        For large-scale usage, consider handling exceptions, retries, rate-limits, etc.
        """
        response = openai.Embedding.create(model=self.model_name, input=text)
        # OpenAI returns: response["data"][0]["embedding"]
        embedding = response["data"][0]["embedding"]
        # Convert the embedding to a standard Python list (it may already be a list, but just to be sure)
        return list(embedding)


class HuggingFaceTextEmbedder(TextEmbedderBase):
    """
    Text embedding using a Hugging Face Transformer model.
    (Requires installing transformers, e.g., `pip install transformers torch`)
    """

    def __init__(self, model_name: str):
        """
        model_name: Hugging Face model name, e.g., "sentence-transformers/all-MiniLM-L6-v2"
        """
        self.model_name = model_name
        # You could load the model/tokenizer here, e.g.:
        # from transformers import AutoTokenizer, AutoModel
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.model = AutoModel.from_pretrained(self.model_name)
        # For brevity, we'll skip the actual loading in this example.

    def embed(self, text: str) -> List[float]:
        """
        Runs text through a Hugging Face model and returns a list of floats.
        """
        # Example (pseudo-code):
        # inputs = self.tokenizer(text, return_tensors="pt")
        # outputs = self.model(**inputs)
        # last_hidden_state = outputs.last_hidden_state
        # pooling to get sentence embedding, e.g., mean pooling:
        # embedding_tensor = last_hidden_state.mean(dim=1).squeeze()
        # embedding = embedding_tensor.detach().numpy().tolist()
        #
        # For demonstration, let's assume we've done it and return a dummy:
        return [0.1, 0.2, 0.3, 0.4]  # Example placeholder


###############################################################################
#                            EMBEDDING FACTORY
###############################################################################


class EmbeddingFactory:
    """
    Factory for creating text embedders based on a config dict.
    Extend this to handle multiple providers (OpenAI, Hugging Face, etc.).
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

        elif provider == "huggingface":
            model_name = config.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            return HuggingFaceTextEmbedder(model_name)

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
