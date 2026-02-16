"""Main entry point: CLI runner for the visual memory pipeline."""

import argparse
import logging
import signal
import sys
import time

import cv2
import numpy as np
import yaml
from PIL import Image

from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload
from src.memory.visual_memory import VisualMemory
from src.navigation.controller import NavigationController
from src.perception.encoder import CLIPEncoder
from src.perception.keyframe_selector import KeyframeSelector
from src.retrieval.change_detector import ChangeDetector
from src.retrieval.retriever import SceneRetriever

logger = logging.getLogger(__name__)

_shutdown = False


def _signal_handler(sig: int, frame: object) -> None:
    global _shutdown
    logger.info("Shutdown signal received")
    _shutdown = True


def load_config(path: str = "config/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(video_path: str, room_id: str, config: dict) -> None:
    """Run the full visual memory pipeline on a video file.

    Args:
        video_path: Path to video file.
        room_id: Room identifier for metadata.
        config: Configuration dictionary.
    """
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Initialize components
    encoder = CLIPEncoder(
        model_name=config["perception"]["model_name"],
        pretrained=config["perception"]["pretrained"],
        device=config["perception"].get("device"),
    )
    selector = KeyframeSelector(threshold=config["keyframe"]["threshold"])

    mem_cfg = config["memory"]
    qdrant = QdrantMemoryClient(
        host=mem_cfg["qdrant_host"],
        port=mem_cfg["qdrant_port"],
        collection_name=mem_cfg["collection_name"],
        vector_size=mem_cfg["vector_size"],
        quantization_config=mem_cfg.get("quantization"),
    )
    memory = VisualMemory(client=qdrant)
    retriever = SceneRetriever(
        client=qdrant,
        top_k=config["retrieval"]["top_k"],
        score_threshold=config["retrieval"]["score_threshold"],
    )
    nav = NavigationController(
        confident_threshold=config["retrieval"]["confident_match"],
        partial_threshold=config["retrieval"]["partial_match"],
    )
    change_detector = ChangeDetector(
        ema_alpha=config["change_detection"]["ema_alpha"],
        change_threshold=config["change_detection"]["change_threshold"],
    )

    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        sys.exit(1)

    frame_count = 0
    keyframe_count = 0

    logger.info("Processing video: %s (room=%s)", video_path, room_id)

    while not _shutdown:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embedding = encoder.encode(image)

        is_kf, emb = selector.is_keyframe(embedding)
        if not is_kf:
            continue

        keyframe_count += 1
        ts = time.time()

        # Store
        payload = MemoryPayload(timestamp=ts, room_id=room_id)
        memory.store(embedding=emb, payload=payload)

        # Retrieve and decide
        results = retriever.query(emb, room_id=room_id)
        decision = nav.decide(results)

        # Change detection
        scores = [r.score for r in results]
        change = change_detector.update(scores)

        logger.info(
            "Frame %d | KF %d | %s (score=%.3f) | changed=%s",
            frame_count,
            keyframe_count,
            decision.action.value,
            decision.top_score,
            change.changed,
        )

    # Flush remaining buffer
    memory.flush()
    cap.release()

    logger.info(
        "Done. Processed %d frames, %d keyframes.", frame_count, keyframe_count
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Robot Visual Memory — process video through visual memory pipeline"
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--room", default="default", help="Room identifier")
    parser.add_argument(
        "--config", default="config/default.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    log_level = config.get("logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_pipeline(args.video, args.room, config)


if __name__ == "__main__":
    main()
