# file: embedding_orchestrator.py

from typing import Dict, Any, List
from embedding import EmbeddingBase  # from your existing embedding.py
from fusion_result import FusionResult  # import the class we just created


class EmbeddingOrchestrator:
    """
    Handles embedding of single-level dictionaries/JSON objects by:
      1) Using a map of field -> embedder type (or some default/inference),
      2) Optionally ignoring specified fields (e.g. user_id) for embedding,
      3) Collecting each field's embedding,
      4) Concatenating them into a final fused embedding,
      5) Returning a FusionResult with metadata.
    """

    def __init__(
        self,
        embedder_map: Dict[str, EmbeddingBase],
        field_type_map: Dict[str, str] = None,
        default_type: str = "text",
        list_no_embed_fields: List[str] = None,
    ):
        """
        :param embedder_map: A dictionary mapping embedder labels to embedder instances,
                             e.g. { "text": <TextEmbedder>, "numeric": <NumericEmbedder>, "category": <CategoryEmbedder> }.
        :param field_type_map: A dictionary mapping field names to embedder labels,
                               e.g. { "description": "text", "price": "numeric" }.
        :param default_type: If a field isn't in field_type_map, default to this embedder label.
        :param list_no_embed_fields: A list of field names for which no embedding should be created.
        """
        self.embedder_map = embedder_map
        self.field_type_map = field_type_map or {}
        self.default_type = default_type
        self.list_no_embed_fields = list_no_embed_fields or []

    def generate_embedding(self, data: Dict[str, Any]) -> FusionResult:
        """
        Generates a FusionResult from a single-level dictionary (database row or JSON).
        Fields listed in self.list_no_embed_fields are ignored (i.e., no embedding is created).
        """
        component_embeddings: Dict[str, List[float]] = {}

        # Process each field in the data that is not in the ignore list.
        for field, value in data.items():
            if field in self.list_no_embed_fields:
                continue

            embed_type = self.field_type_map.get(field, self.default_type)
            embedder = self.embedder_map.get(embed_type)
            if not embedder:
                raise ValueError(f"No embedder found for type '{embed_type}'")

            field_embedding = embedder.embed(value)
            component_embeddings[field] = field_embedding

        # Concatenate embeddings in the order of fields (skipping ignored ones)
        fusion_embedding: List[float] = []
        for field in data.keys():
            if field in self.list_no_embed_fields:
                continue
            fusion_embedding.extend(component_embeddings[field])

        return FusionResult(
            original_data=data,
            component_embeddings=component_embeddings,
            fusion_embedding=fusion_embedding,
        )
