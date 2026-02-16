"""Run memory compression on old entries."""

import argparse
import logging

from src.main import load_config
from src.memory.compressor import MemoryCompressor
from src.memory.qdrant_client import QdrantMemoryClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress old visual memories")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    mem_cfg = config["memory"]

    qdrant = QdrantMemoryClient(
        host=mem_cfg["qdrant_host"],
        port=mem_cfg["qdrant_port"],
        collection_name=mem_cfg["collection_name"],
        vector_size=mem_cfg["vector_size"],
    )

    comp_cfg = config["compression"]
    compressor = MemoryCompressor(
        client=qdrant,
        keep_every_nth=comp_cfg["keep_every_nth"],
        age_threshold_hours=comp_cfg["age_threshold_hours"],
    )

    deleted = compressor.compress()
    logger.info("Compression complete: %d points removed", deleted)


if __name__ == "__main__":
    main()
