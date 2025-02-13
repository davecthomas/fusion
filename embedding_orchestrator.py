# file: embedding_orchestrator.py

from typing import Dict, Any, List
from embedding import EmbeddingBase  # from your existing embedding.py
from fusion_result import FusionResult  # import the class we just created


class EmbeddingOrchestrator:
    """
    Handles embedding of single-level dictionaries/JSON objects by:
      1) Using a map of field -> embedder type (or some default/inference),
      2) Collecting each field's embedding,
      3) Concatenating them into a final fused embedding,
      4) Returning a FusionResult with metadata.
    """

    def __init__(
        self,
        embedder_map: Dict[str, EmbeddingBase],
        field_type_map: Dict[str, str] = None,
        default_type: str = "text",
    ):
        """
        :param embedder_map: e.g. { "text": <TextEmbedder>, "numeric": <NumericEmbedder>, "category": <CategoryEmbedder> }
        :param field_type_map: A dict of field_name -> type_label (e.g., { "description": "text", "price": "numeric" })
        :param default_type: If a field isn't in field_type_map, we can default to "text" or some other embedder type.
        """
        self.embedder_map = embedder_map
        self.field_type_map = field_type_map or {}
        self.default_type = default_type

    def generate_embedding(self, data: Dict[str, Any]) -> FusionResult:
        """
        Generates a FusionResult from a single-level dict (database row or JSON).
        """
        component_embeddings: Dict[str, List[float]] = {}

        # 1) For each field, determine the embedder and embed its value
        for field, value in data.items():
            embed_type = self.field_type_map.get(field, self.default_type)
            embedder = self.embedder_map.get(embed_type)
            if not embedder:
                raise ValueError(f"No embedder found for type '{embed_type}'")

            field_embedding = embedder.embed(value)
            component_embeddings[field] = field_embedding

        # 2) Concatenate in the order of the fields to form the fused vector
        fusion_embedding = []
        for field in data.keys():
            fusion_embedding.extend(component_embeddings[field])

        # 3) Build a FusionResult object
        return FusionResult(
            original_data=data,
            component_embeddings=component_embeddings,
            fusion_embedding=fusion_embedding,
        )
