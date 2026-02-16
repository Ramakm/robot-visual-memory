"""Reset (delete and recreate) the Qdrant collection."""

import argparse
import logging

from src.main import load_config
from src.memory.qdrant_client import QdrantMemoryClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the visual memory collection")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    config = load_config(args.config)
    mem_cfg = config["memory"]

    if not args.yes:
        name = mem_cfg["collection_name"]
        answer = input(f"Delete collection '{name}'? [y/N] ")
        if answer.lower() != "y":
            print("Aborted.")
            return

    qdrant = QdrantMemoryClient(
        host=mem_cfg["qdrant_host"],
        port=mem_cfg["qdrant_port"],
        collection_name=mem_cfg["collection_name"],
        vector_size=mem_cfg["vector_size"],
    )
    qdrant.delete_collection()
    logger.info("Collection reset complete")


if __name__ == "__main__":
    main()
