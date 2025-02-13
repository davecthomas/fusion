# file: main.py

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any

from appconfig import AppConfig

# from database import TheDatabase  # Omitted from this local test scenario
from embedding import (
    EmbeddingFactory,
    NumericEmbedder,
    CategoryEmbedder,
)
from embedding_orchestrator import EmbeddingOrchestrator
from fusion_result import FusionResult
from similarity import Similarity  # <-- Importing our new Similarity class

###############################################################################
#                         ENUM & USERSTATE CLASS
###############################################################################


class FunnelStage(Enum):
    NOT_CLAIMED = "not_claimed"
    CLAIMED_OFFER = "claimed_offer"
    CLAIMED_OFFER_COMPLETED = "claimed_offer_completed"
    INACTIVE_NOT_CLAIMED = "inactive_not_claimed"


@dataclass
class UserState:
    """
    Represents a user in a particular funnel state, with numeric and text fields.
    """

    user_id: str
    funnel_stage: FunnelStage
    num_offers_claimed: int
    num_offers_completed: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the UserState into a standard Python dict.
        Enums become their string values for category embedding.
        """
        dict_data = asdict(self)
        dict_data["funnel_stage"] = self.funnel_stage.value
        return dict_data


###############################################################################
#                           MAIN TEST LOGIC
###############################################################################


def main():
    # 1) Load environment config (useful for actual text embeddings with OpenAI/Hugging Face).
    ac_config = AppConfig(".env")

    # 2) Create 10 UserState objects with varied numeric and funnel states
    list_us_user_states = [
        UserState("user001", FunnelStage.NOT_CLAIMED, 0, 0),
        UserState("user002", FunnelStage.CLAIMED_OFFER, 1, 0),
        UserState("user003", FunnelStage.CLAIMED_OFFER, 2, 0),
        UserState("user004", FunnelStage.CLAIMED_OFFER_COMPLETED, 1, 1),
        UserState("user005", FunnelStage.CLAIMED_OFFER_COMPLETED, 2, 2),
        UserState("user006", FunnelStage.CLAIMED_OFFER_COMPLETED, 2, 1),
        UserState("user007", FunnelStage.CLAIMED_OFFER_COMPLETED, 3, 3),
        UserState("user008", FunnelStage.INACTIVE_NOT_CLAIMED, 0, 0),
        UserState("user009", FunnelStage.NOT_CLAIMED, 0, 0),
        UserState("user010", FunnelStage.CLAIMED_OFFER_COMPLETED, 1, 1),
    ]

    # 3) Compute min/max for numeric fields across all user states
    list_num_offers_claimed_vals = [
        us_state.num_offers_claimed for us_state in list_us_user_states
    ]
    list_num_offers_completed_vals = [
        us_state.num_offers_completed for us_state in list_us_user_states
    ]

    num_offers_claimed_min, num_offers_claimed_max = (
        min(list_num_offers_claimed_vals),
        max(list_num_offers_claimed_vals),
    )
    num_offers_completed_min, num_offers_completed_max = (
        min(list_num_offers_completed_vals),
        max(list_num_offers_completed_vals),
    )

    # 4) Build text embedder config (OpenAI or Hugging Face)
    dict_text_embed_config = {
        "provider": ac_config.OPENAI_PROVIDER,  # "openai" or "huggingface"
        "api_key": ac_config.OPENAI_API_KEY,
        "model_name": ac_config.OPENAI_MODEL_NAME,
    }
    te_text_embedder = EmbeddingFactory.create_text_embedder(dict_text_embed_config)

    # 5) Instantiate numeric & category embedders
    ne_offers_claimed_embedder = NumericEmbedder(
        num_offers_claimed_min, num_offers_claimed_max
    )
    ne_offers_completed_embedder = NumericEmbedder(
        num_offers_completed_min, num_offers_completed_max
    )

    list_funnel_stage_cats = [
        FunnelStage.NOT_CLAIMED.value,
        FunnelStage.CLAIMED_OFFER.value,
        FunnelStage.CLAIMED_OFFER_COMPLETED.value,
        FunnelStage.INACTIVE_NOT_CLAIMED.value,
    ]
    ce_funnel_stage_embedder = CategoryEmbedder(list_funnel_stage_cats)

    # 6) Map embedder labels to actual embedders
    dict_embedder_map = {
        "text": te_text_embedder,
        "offers_claimed_num": ne_offers_claimed_embedder,
        "offers_completed_num": ne_offers_completed_embedder,
        "funnel_stage_cat": ce_funnel_stage_embedder,
    }

    # 7) Field → embedder label
    dict_field_type_map = {
        "user_id": "text",
        "funnel_stage": "funnel_stage_cat",
        "num_offers_claimed": "offers_claimed_num",
        "num_offers_completed": "offers_completed_num",
    }

    # 8) Create the EmbeddingOrchestrator
    eo_orchestrator = EmbeddingOrchestrator(
        embedder_map=dict_embedder_map,
        field_type_map=dict_field_type_map,
        default_type="text",
    )

    # 9) Generate FusionResult for each UserState
    list_fr_results: List[FusionResult] = []
    for idx, us_state in enumerate(list_us_user_states, start=1):
        dict_user_data = us_state.to_dict()
        fr_result = eo_orchestrator.generate_embedding(dict_user_data)
        list_fr_results.append(fr_result)

        print(f"--- UserState #{idx} ---")
        print(f"Original Data: {fr_result.original_data}")
        print(f"Component Embeddings: {fr_result.component_embeddings}")
        print(
            f"Fused Embedding (len={len(fr_result.fusion_embedding)}): {fr_result.fusion_embedding}"
        )
        print()

    # 10) Demonstrate vector similarity comparisons between two or more fusion vectors
    sim = Similarity()

    # Example: Compare the first two user states if they exist
    if len(list_fr_results) >= 2:
        fr_a = list_fr_results[0]
        fr_b = list_fr_results[1]

        dist_euclid = sim.calc_euclidean(fr_a.fusion_embedding, fr_b.fusion_embedding)
        dist_manhattan = sim.calc_manhattan(
            fr_a.fusion_embedding, fr_b.fusion_embedding
        )
        sim_cosine = sim.calc_cosine(fr_a.fusion_embedding, fr_b.fusion_embedding)

        print("--- Similarity Comparison (UserState #1 vs. #2) ---")
        print(f"Euclidean Distance: {dist_euclid}")
        print(f"Manhattan Distance: {dist_manhattan}")
        print(f"Cosine Similarity:  {sim_cosine}")


if __name__ == "__main__":
    main()
