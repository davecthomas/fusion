# file: main.py

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any

import pandas as pd

from appconfig import AppConfig

from database import TheDatabase
from embedding import (
    EmbeddingFactory,
    NumericEmbedder,
    CategoryEmbedder,
    BooleanEmbedder,
)
from embedding_orchestrator import EmbeddingOrchestrator
from fusion_result import FusionResult
from similarity import Similarity

###############################################################################
#                         ENUM & USERSTATE CLASS
###############################################################################


from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any


class FunnelStage(Enum):
    NEVER_CLAIMED = "never_claimed"
    CLAIMED_OFFER = "claimed_offer"
    CLAIM_EXPIRED = "claim_expired"
    TRANSACTION_COMPLETED = "transaction_completed"
    CASH_BACK_COMPLETED = "cash_back_completed"
    CASH_BACK_REJECTED = "cash_back_rejected"
    INACTIVE_ONE_MONTH = "inactive_one_month"


@dataclass
class UserState:
    """
    Represents a user in a particular funnel state, with numeric, float, and text fields.
    """

    user_id: str
    user_source: str
    funnel_stage: FunnelStage
    num_offers_claimed: int
    num_offers_completed: int
    cash_back_balance: float
    cash_back_redeemed: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this UserState into a standard dict.
        Enums become their string values for category embedding.
        """
        dict_data = asdict(self)
        dict_data["funnel_stage"] = self.funnel_stage.value
        return dict_data


###############################################################################
#                           MAIN TEST LOGIC
###############################################################################
import pandas as pd
from similarity import Similarity


def compute_similarity_dataframes(list_fr_results, id_key: str = "user_id"):
    """
    Given a list of FusionResult objects, computes pairwise Euclidean distance,
    Manhattan distance, and Cosine similarity. Returns a tuple of 4 dataframes:
      (df_euclid, df_manhattan, df_cosine, df_all)

    :param list_fr_results: A list of FusionResult objects
    :param id_key: The key in original_data used to identify each record (default 'user_id').
                   If missing, we fall back to 'record_{index}'.

    Example usage:
        df_euclid, df_manhattan, df_cosine, df_all = compute_similarity_dataframes(results)
    """
    sim = Similarity()

    list_dict_euclid = []
    list_dict_manhattan = []
    list_dict_cosine = []

    # 1) Pairwise similarity for i < j
    for i in range(len(list_fr_results)):
        for j in range(i + 1, len(list_fr_results)):
            fr_i = list_fr_results[i]
            fr_j = list_fr_results[j]

            # Use user_id or a fallback
            record_id_i = fr_i.original_data.get(id_key, f"record_{i+1}")
            record_id_j = fr_j.original_data.get(id_key, f"record_{j+1}")

            vec_i = fr_i.fusion_embedding
            vec_j = fr_j.fusion_embedding

            # Calculate distances/similarities
            dist_euclid = sim.calc_euclidean(vec_i, vec_j)
            dist_manhattan = sim.calc_manhattan(vec_i, vec_j)
            sim_cosine_val = sim.calc_cosine(vec_i, vec_j)

            # Append to each list
            list_dict_euclid.append(
                {
                    "record_a": record_id_i,
                    "record_b": record_id_j,
                    "euclidean": dist_euclid,
                }
            )
            list_dict_manhattan.append(
                {
                    "record_a": record_id_i,
                    "record_b": record_id_j,
                    "manhattan": dist_manhattan,
                }
            )
            list_dict_cosine.append(
                {
                    "record_a": record_id_i,
                    "record_b": record_id_j,
                    "cosine_similarity": sim_cosine_val,
                }
            )

    # 2) Create DataFrames
    df_euclid = pd.DataFrame(list_dict_euclid)
    df_manhattan = pd.DataFrame(list_dict_manhattan)
    df_cosine = pd.DataFrame(list_dict_cosine)

    # 3) Normalize distances (optional) and rename for clarity
    if not df_euclid.empty:
        max_euclid = df_euclid["euclidean"].max()
        df_euclid["euclid_score"] = 1 - (df_euclid["euclidean"] / max_euclid)

    if not df_manhattan.empty:
        max_manhattan = df_manhattan["manhattan"].max()
        df_manhattan["manhattan_score"] = 1 - (
            df_manhattan["manhattan"] / max_manhattan
        )

    if not df_cosine.empty:
        # Cosine similarity in [-1, 1] → [0, 1]
        df_cosine["cosine_score"] = (df_cosine["cosine_similarity"] + 1) / 2.0

    # 4) Merge into a single combined dataframe
    df_merged = df_euclid.merge(
        df_manhattan[["record_a", "record_b", "manhattan", "manhattan_score"]],
        on=["record_a", "record_b"],
        how="outer",
    )
    df_all = df_merged.merge(
        df_cosine[["record_a", "record_b", "cosine_similarity", "cosine_score"]],
        on=["record_a", "record_b"],
        how="outer",
    )

    # Return all 4 DataFrames so the caller can save or inspect as needed
    return (df_euclid, df_manhattan, df_cosine, df_all)


def test_dummy_user_state_dictionaries():
    # 1) Load environment config (useful for actual text embeddings with OpenAI/Hugging Face).
    ac_config = AppConfig(".env")

    list_user_sources: List[str] = [
        "Referral",
        "Organic",
        "Radio",
        "Uber",
        "Partner B",
        "Apple",
        "Applovin",
    ]

    # Create 30 user states using explicit assignments with list_user_sources indices
    list_us_user_states: List[UserState] = [
        UserState(
            "user001", list_user_sources[0], FunnelStage.NEVER_CLAIMED, 0, 0, 0.00, 0.00
        ),
        UserState(
            "user002", list_user_sources[1], FunnelStage.CLAIMED_OFFER, 1, 0, 5.99, 0.00
        ),
        UserState(
            "user003",
            list_user_sources[2],
            FunnelStage.CLAIM_EXPIRED,
            2,
            0,
            25.75,
            10.00,
        ),
        UserState(
            "user004",
            list_user_sources[3],
            FunnelStage.TRANSACTION_COMPLETED,
            1,
            1,
            13.35,
            5.00,
        ),
        UserState(
            "user005",
            list_user_sources[4],
            FunnelStage.CASH_BACK_COMPLETED,
            2,
            2,
            34.90,
            20.00,
        ),
        UserState(
            "user006",
            list_user_sources[5],
            FunnelStage.CASH_BACK_REJECTED,
            2,
            1,
            45.00,
            25.00,
        ),
        UserState(
            "user007",
            list_user_sources[6],
            FunnelStage.INACTIVE_ONE_MONTH,
            0,
            0,
            0.00,
            0.00,
        ),
        UserState(
            "user008", list_user_sources[0], FunnelStage.NEVER_CLAIMED, 0, 0, 0.00, 0.00
        ),
        UserState(
            "user009",
            list_user_sources[1],
            FunnelStage.CLAIMED_OFFER,
            3,
            1,
            65.99,
            30.00,
        ),
        UserState(
            "user010",
            list_user_sources[2],
            FunnelStage.CASH_BACK_COMPLETED,
            4,
            4,
            75.00,
            10.00,
        ),
        UserState(
            "user011",
            list_user_sources[3],
            FunnelStage.CLAIM_EXPIRED,
            2,
            0,
            15.21,
            0.00,
        ),
        UserState(
            "user012",
            list_user_sources[4],
            FunnelStage.TRANSACTION_COMPLETED,
            4,
            2,
            7.75,
            5.00,
        ),
        UserState(
            "user013",
            list_user_sources[5],
            FunnelStage.CASH_BACK_REJECTED,
            5,
            3,
            28.49,
            15.25,
        ),
        UserState(
            "user014",
            list_user_sources[6],
            FunnelStage.INACTIVE_ONE_MONTH,
            0,
            0,
            0.00,
            0.00,
        ),
        UserState(
            "user015", list_user_sources[0], FunnelStage.CLAIMED_OFFER, 2, 1, 6.77, 2.12
        ),
        UserState(
            "user016",
            list_user_sources[1],
            FunnelStage.CLAIM_EXPIRED,
            2,
            0,
            11.11,
            0.00,
        ),
        UserState(
            "user017",
            list_user_sources[2],
            FunnelStage.CASH_BACK_COMPLETED,
            6,
            6,
            75.00,
            75.00,
        ),
        UserState(
            "user018",
            list_user_sources[3],
            FunnelStage.CASH_BACK_REJECTED,
            4,
            4,
            32.56,
            16.25,
        ),
        UserState(
            "user019", list_user_sources[4], FunnelStage.NEVER_CLAIMED, 0, 0, 0.00, 0.00
        ),
        UserState(
            "user020",
            list_user_sources[5],
            FunnelStage.INACTIVE_ONE_MONTH,
            0,
            0,
            9.99,
            0.00,
        ),
        UserState(
            "user021",
            list_user_sources[6],
            FunnelStage.TRANSACTION_COMPLETED,
            3,
            3,
            44.44,
            11.11,
        ),
        UserState(
            "user022",
            list_user_sources[0],
            FunnelStage.CASH_BACK_COMPLETED,
            5,
            5,
            70.00,
            10.00,
        ),
        UserState(
            "user023",
            list_user_sources[1],
            FunnelStage.CLAIMED_OFFER,
            3,
            2,
            16.00,
            5.99,
        ),
        UserState(
            "user024",
            list_user_sources[2],
            FunnelStage.CASH_BACK_REJECTED,
            5,
            4,
            63.25,
            20.00,
        ),
        UserState(
            "user025", list_user_sources[3], FunnelStage.CLAIM_EXPIRED, 1, 0, 5.00, 0.00
        ),
        UserState(
            "user026",
            list_user_sources[4],
            FunnelStage.CLAIMED_OFFER,
            2,
            0,
            10.50,
            0.00,
        ),
        UserState(
            "user027",
            list_user_sources[5],
            FunnelStage.TRANSACTION_COMPLETED,
            7,
            7,
            40.25,
            30.25,
        ),
        UserState(
            "user028",
            list_user_sources[6],
            FunnelStage.CASH_BACK_COMPLETED,
            8,
            8,
            75.00,
            40.00,
        ),
        UserState(
            "user029",
            list_user_sources[0],
            FunnelStage.CASH_BACK_REJECTED,
            6,
            5,
            20.99,
            10.00,
        ),
        UserState(
            "user030",
            list_user_sources[1],
            FunnelStage.INACTIVE_ONE_MONTH,
            1,
            0,
            0.00,
            0.00,
        ),
    ]

    # 3) Gather min/max for each numeric or float field
    list_num_offers_claimed_vals = [
        us_state.num_offers_claimed for us_state in list_us_user_states
    ]
    list_num_offers_completed_vals = [
        us_state.num_offers_completed for us_state in list_us_user_states
    ]
    list_cash_back_balance_vals = [
        us_state.cash_back_balance for us_state in list_us_user_states
    ]
    list_cash_back_redeemed_vals = [
        us_state.cash_back_redeemed for us_state in list_us_user_states
    ]

    num_offers_claimed_min, num_offers_claimed_max = (
        min(list_num_offers_claimed_vals),
        max(list_num_offers_claimed_vals),
    )
    num_offers_completed_min, num_offers_completed_max = (
        min(list_num_offers_completed_vals),
        max(list_num_offers_completed_vals),
    )
    cash_back_balance_min, cash_back_balance_max = (
        min(list_cash_back_balance_vals),
        max(list_cash_back_balance_vals),
    )
    cash_back_redeemed_min, cash_back_redeemed_max = (
        min(list_cash_back_redeemed_vals),
        max(list_cash_back_redeemed_vals),
    )

    # 4) Build text embedder config
    dict_text_embed_config = {
        "provider": ac_config.OPENAI_PROVIDER,  # "openai", ...
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
    ne_cash_back_balance_embedder = NumericEmbedder(
        cash_back_balance_min, cash_back_balance_max
    )
    ne_cash_back_redeemed_embedder = NumericEmbedder(
        cash_back_redeemed_min, cash_back_redeemed_max
    )

    list_funnel_stage_cats = [
        FunnelStage.NEVER_CLAIMED.value,
        FunnelStage.CLAIMED_OFFER.value,
        FunnelStage.CLAIM_EXPIRED.value,
        FunnelStage.TRANSACTION_COMPLETED.value,
        FunnelStage.CASH_BACK_COMPLETED.value,
        FunnelStage.CASH_BACK_REJECTED.value,
        FunnelStage.INACTIVE_ONE_MONTH.value,
    ]
    ce_funnel_stage_embedder = CategoryEmbedder(list_funnel_stage_cats)
    # Create a category embedder for user_source
    ce_user_source_embedder = CategoryEmbedder(list_user_sources)

    dict_embedder_map = {
        "user_source_cat": ce_user_source_embedder,
        "offers_claimed_num": ne_offers_claimed_embedder,
        "offers_completed_num": ne_offers_completed_embedder,
        "cash_back_balance_num": ne_cash_back_balance_embedder,
        "cash_back_redeemed_num": ne_cash_back_redeemed_embedder,
        "funnel_stage_cat": ce_funnel_stage_embedder,
        "text": te_text_embedder,
    }

    # 8) Field -> embedder label. Note: we ignore "user_id", but we do embed "user_source"
    dict_field_type_map = {
        "user_source": "user_source_cat",
        "funnel_stage": "funnel_stage_cat",
        "num_offers_claimed": "offers_claimed_num",
        "num_offers_completed": "offers_completed_num",
        "cash_back_balance": "cash_back_balance_num",
        "cash_back_redeemed": "cash_back_redeemed_num",
    }

    # 8) Create the EmbeddingOrchestrator with a list of fields to ignore (e.g., "user_id")
    eo_orchestrator = EmbeddingOrchestrator(
        embedder_map=dict_embedder_map,
        field_type_map=dict_field_type_map,
        default_type="text",
        list_no_embed_fields=["user_id"],
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

    (df_euclid, df_manhattan, df_cosine, df_all) = compute_similarity_dataframes(
        list_fr_results,
        id_key="user_id",  # or the field you want to use as an identifier
    )

    # Save the merged dataframe to CSV
    df_all.to_csv("combined_similarity.csv", index=False)


def test_database():
    import pandas as pd
    from similarity import Similarity

    config = AppConfig()
    db = TheDatabase(config)

    # Get schema info for dynamic inclusion
    schema_info = db.get_schema_info(None)
    included_cols = list(schema_info.keys())

    # Exclude certain columns from embedding
    excluded_from_embedding = ["USER_UUID"]

    # Fetch limited rows
    records = db.get_records(None, 10)
    filtered_records = [
        {k: v for k, v in row.items() if k in included_cols} for row in records
    ]

    # Instantiate embedders, including the new BooleanEmbedder
    numeric_embedder = NumericEmbedder(0, 100)
    category_embedder = CategoryEmbedder(["sample_cat_1", "sample_cat_2"])
    text_embedder = EmbeddingFactory.create_text_embedder(
        {
            "provider": config.OPENAI_PROVIDER,
            "api_key": config.OPENAI_API_KEY,
            "model_name": config.OPENAI_MODEL_NAME,
        }
    )

    from embedding import BooleanEmbedder

    bool_embedder = BooleanEmbedder()

    # Build an embedder map
    embedder_map = {
        "numeric": numeric_embedder,
        "category": category_embedder,
        "text": text_embedder,
        "bool": bool_embedder,
    }

    # Build field_type_map from schema, based on Snowflake data types
    field_type_map = {}
    for col_name, data_type in schema_info.items():
        data_type = data_type.upper()

        # 1) Numeric
        if any(x in data_type for x in ["NUMBER", "INT", "FLOAT", "DECIMAL"]):
            field_type_map[col_name] = "numeric"

        # 2) Text
        elif any(x in data_type for x in ["VARCHAR", "TEXT", "CHAR"]):
            field_type_map[col_name] = "text"

        # 3) Boolean
        elif "BOOLEAN" in data_type:
            field_type_map[col_name] = "bool"

        # 4) Fallback to text by default
        else:
            field_type_map[col_name] = "text"

    orchestrator = EmbeddingOrchestrator(
        embedder_map=embedder_map,
        field_type_map=field_type_map,
        default_type="text",
        list_no_embed_fields=excluded_from_embedding,
    )

    # 1) Embed each record and store the FusionResult in a list
    list_fr_results = []
    for row_data in filtered_records:
        result = orchestrator.generate_embedding(row_data)
        list_fr_results.append(result)
        print("Original:", result.original_data)
        print("Embeddings:", result.component_embeddings)
        print("Fusion:", result.fusion_embedding)
        print()

    (df_euclid, df_manhattan, df_cosine, df_all) = compute_similarity_dataframes(
        list_fr_results,
        id_key="USER_UUID",  # or the field you want to use as an identifier
    )

    # 5) Save to CSV
    df_all.to_csv("db_combined_similarity.csv", index=False)
    print("\nSimilarity metrics saved to db_combined_similarity.csv\n")


if __name__ == "__main__":
    # test_dummy_user_state_dictionaries()
    test_database()
