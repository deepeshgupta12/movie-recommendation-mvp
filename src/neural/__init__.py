"""
Neural retrieval layer (V3).

This package will add:
- Two-tower embedding training
- Embedding export for users/items
- ANN index build (HNSW)
- Neural candidate retrieval API

Design principle:
Keep all functions import-safe and contract-first.
Heavy compute is done only inside explicit main() entrypoints.
"""