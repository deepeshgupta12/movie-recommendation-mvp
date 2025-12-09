from __future__ import annotations

from pathlib import Path

from src.config.settings import settings

PROCESSED: Path = settings.PROCESSED_DIR

# Two-tower data
TT_TRAIN = PROCESSED / "two_tower_train.parquet"
TT_VAL = PROCESSED / "two_tower_val.parquet"

# Embeddings
USER_EMB = PROCESSED / "user_emb_v3.parquet"
ITEM_EMB = PROCESSED / "item_emb_v3.parquet"

# ANN index
HNSW_INDEX = PROCESSED / "item_hnsw_v3.index"
HNSW_META = PROCESSED / "item_hnsw_v3.meta.json"

# Sequence data
SEQ_TRAIN = PROCESSED / "user_seq_train.parquet"
SEQ_VAL = PROCESSED / "user_seq_val.parquet"