# file: fusion_result.py

from typing import Any, Dict, List


class FusionResult:
    """
    Holds the metadata and the fused embedding for a single data row (dict/JSON).
    """

    def __init__(
        self,
        original_data: Dict[str, Any],
        component_embeddings: Dict[str, List[float]],
        fusion_embedding: List[float],
    ):
        """
        :param original_data: The input dict/JSON that was embedded.
        :param component_embeddings: Map of each key -> its embedding (list of floats).
        :param fusion_embedding: The final fused embedding (list of floats).
        """
        self.original_data = original_data
        self.component_embeddings = component_embeddings
        self.fusion_embedding = fusion_embedding

    def __repr__(self) -> str:
        return (
            f"FusionResult(\n"
            f"  original_data={self.original_data},\n"
            f"  component_embeddings={self.component_embeddings},\n"
            f"  fusion_embedding={self.fusion_embedding}\n"
            f")"
        )
