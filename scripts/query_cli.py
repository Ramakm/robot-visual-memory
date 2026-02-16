"""CLI tool to query visual memory with an image."""

import argparse
import logging

from PIL import Image

from src.main import load_config
from src.memory.qdrant_client import QdrantMemoryClient
from src.navigation.controller import NavigationController
from src.perception.encoder import CLIPEncoder
from src.retrieval.retriever import SceneRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query visual memory with an image")
    parser.add_argument("--image", required=True, help="Path to query image")
    parser.add_argument("--room", default=None, help="Filter by room")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

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
    retriever = SceneRetriever(
        client=qdrant,
        top_k=args.top_k,
        score_threshold=config["retrieval"]["score_threshold"],
    )
    nav = NavigationController(
        confident_threshold=config["retrieval"]["confident_match"],
        partial_threshold=config["retrieval"]["partial_match"],
    )

    image = Image.open(args.image).convert("RGB")
    embedding = encoder.encode(image)

    results = retriever.query(embedding, room_id=args.room, top_k=args.top_k)
    decision = nav.decide(results)

    print(f"\nNavigation Decision: {decision.action.value}")
    print(f"Confidence: {decision.confidence:.3f}")
    print(f"Matches: {decision.num_matches}")
    print(f"Room: {decision.room_id}\n")

    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r.score:.4f} room={r.payload.room_id} ts={r.payload.timestamp:.0f}")


if __name__ == "__main__":
    main()
