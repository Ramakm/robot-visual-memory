"""Performance scaling test: simulate 10k vectors and measure behavior."""

import logging
import time

import numpy as np

from src.main import load_config
from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload
from src.memory.visual_memory import VisualMemory
from src.retrieval.retriever import SceneRetriever
from src.utils.timing import LatencyTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOTAL_VECTORS = 10_000
BATCH_SIZE = 100
QUERY_INTERVAL = 1000


def main() -> None:
    config = load_config("config/benchmark.yaml")
    mem_cfg = config["memory"]

    qdrant = QdrantMemoryClient(
        host=mem_cfg["qdrant_host"],
        port=mem_cfg["qdrant_port"],
        collection_name=mem_cfg["collection_name"] + "_perf",
        vector_size=mem_cfg["vector_size"],
    )
    memory = VisualMemory(client=qdrant, batch_size=BATCH_SIZE)
    retriever = SceneRetriever(client=qdrant, top_k=5, score_threshold=0.0)

    insert_tracker = LatencyTracker()
    query_tracker = LatencyTracker()

    print(f"\nInserting {TOTAL_VECTORS} vectors...")

    for i in range(TOTAL_VECTORS):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        payload = MemoryPayload(
            timestamp=time.time(),
            room_id=f"room_{i % 10}",
        )

        t0 = time.perf_counter()
        memory.store(embedding=emb, payload=payload)
        insert_tracker.record((time.perf_counter() - t0) * 1000)

        if (i + 1) % QUERY_INTERVAL == 0:
            memory.flush()
            query_emb = np.random.randn(512).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)

            t0 = time.perf_counter()
            results = retriever.query(query_emb)
            query_tracker.record((time.perf_counter() - t0) * 1000)

            print(f"  [{i+1}/{TOTAL_VECTORS}] query_time={query_tracker.mean_ms:.2f}ms count={qdrant.count()}")

    memory.flush()

    print("\n" + "=" * 50)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 50)
    print(f"Total vectors inserted: {TOTAL_VECTORS}")
    print(f"Insert (buffered): mean={insert_tracker.mean_ms:.4f}ms")
    print(f"Query latency: mean={query_tracker.mean_ms:.2f}ms min={query_tracker.min_ms:.2f}ms max={query_tracker.max_ms:.2f}ms")
    print("=" * 50)

    # Cleanup
    qdrant.delete_collection()


if __name__ == "__main__":
    main()
