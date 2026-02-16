"""Ingest a video file into visual memory."""

import argparse
import logging
import time

import cv2
from PIL import Image

from src.main import load_config
from src.memory.qdrant_client import QdrantMemoryClient
from src.memory.schemas import MemoryPayload
from src.memory.visual_memory import VisualMemory
from src.perception.encoder import CLIPEncoder
from src.perception.keyframe_selector import KeyframeSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest video into visual memory")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--room", default="default", help="Room identifier")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

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
    )
    memory = VisualMemory(client=qdrant)

    cap = cv2.VideoCapture(args.video)
    frame_count = 0
    keyframe_count = 0

    while True:
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
        payload = MemoryPayload(timestamp=time.time(), room_id=args.room)
        memory.store(embedding=emb, payload=payload)

        if frame_count % 100 == 0:
            logger.info("Processed %d frames, %d keyframes", frame_count, keyframe_count)

    memory.flush()
    cap.release()
    logger.info("Ingestion complete: %d frames, %d keyframes stored", frame_count, keyframe_count)


if __name__ == "__main__":
    main()
