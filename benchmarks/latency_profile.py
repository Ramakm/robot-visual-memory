"""Latency profiling for core operations: CLIP encode, insert, query."""

import logging
import time

import numpy as np
from PIL import Image

from src.main import load_config
from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload
from src.memory.visual_memory import VisualMemory
from src.perception.encoder import CLIPEncoder
from src.retrieval.retriever import SceneRetriever
from src.utils.timing import LatencyTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_ITERATIONS = 50


def main() -> None:
    config = load_config("config/benchmark.yaml")

    encoder = CLIPEncoder(
        model_name=config["perception"]["model_name"],
        pretrained=config["perception"]["pretrained"],
        device=config["perception"].get("device"),
    )

    mem_cfg = config["memory"]
    qdrant = QdrantMemoryClient(
        host=mem_cfg["qdrant_host"],
        port=mem_cfg["qdrant_port"],
        collection_name=mem_cfg["collection_name"],
        vector_size=mem_cfg["vector_size"],
    )
    memory = VisualMemory(client=qdrant, batch_size=1)
    retriever = SceneRetriever(client=qdrant, top_k=5, score_threshold=0.0)

    encode_tracker = LatencyTracker()
    insert_tracker = LatencyTracker()
    query_tracker = LatencyTracker()

    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    for i in range(NUM_ITERATIONS):
        # Encode
        t0 = time.perf_counter()
        emb = encoder.encode(dummy_image)
        encode_tracker.record((time.perf_counter() - t0) * 1000)

        # Insert
        payload = MemoryPayload(timestamp=time.time(), room_id="bench")
        t0 = time.perf_counter()
        memory.store_immediate(embedding=emb, payload=payload)
        insert_tracker.record((time.perf_counter() - t0) * 1000)

        # Query
        t0 = time.perf_counter()
        retriever.query(emb)
        query_tracker.record((time.perf_counter() - t0) * 1000)

    print("\n" + "=" * 50)
    print("LATENCY PROFILE RESULTS")
    print("=" * 50)
    print(f"{'Operation':<20} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10} {'Count':>8}")
    print("-" * 58)
    for name, tracker in [("CLIP Encode", encode_tracker), ("Qdrant Insert", insert_tracker), ("Qdrant Query", query_tracker)]:
        s = tracker.summary()
        print(f"{name:<20} {s['mean_ms']:>10.2f} {s['min_ms']:>10.2f} {s['max_ms']:>10.2f} {s['count']:>8}")
    print("=" * 50)

    # Cleanup
    qdrant.delete_collection()


if __name__ == "__main__":
    main()
