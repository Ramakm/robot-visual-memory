"""ROS2 node for real-time visual memory integration.

Only importable when rclpy is available. The core pipeline modules
have no ROS dependency.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ROSImage
    from std_msgs.msg import String

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.info("ROS2 not available — node cannot be launched")

if ROS_AVAILABLE:
    from cv_bridge import CvBridge
    from PIL import Image

    from src.memory.qdrant_client import QdrantMemoryClient
    from src.memory.schemas import MemoryPayload
    from src.memory.visual_memory import VisualMemory
    from src.navigation.controller import NavigationController
    from src.perception.encoder import CLIPEncoder
    from src.perception.keyframe_selector import KeyframeSelector
    from src.retrieval.retriever import SceneRetriever
    from src.main import load_config

    class VisualMemoryNode(Node):
        """ROS2 node that subscribes to camera images and publishes navigation decisions."""

        def __init__(self) -> None:
            super().__init__("visual_memory_node")

            self.declare_parameter("config_path", "config/default.yaml")
            self.declare_parameter("room_id", "default")

            config_path = self.get_parameter("config_path").value
            self.room_id = self.get_parameter("room_id").value
            config = load_config(config_path)

            self.encoder = CLIPEncoder(
                model_name=config["perception"]["model_name"],
                pretrained=config["perception"]["pretrained"],
                device=config["perception"].get("device"),
            )
            self.selector = KeyframeSelector(
                threshold=config["keyframe"]["threshold"]
            )

            mem_cfg = config["memory"]
            qdrant = QdrantMemoryClient(
                host=mem_cfg["qdrant_host"],
                port=mem_cfg["qdrant_port"],
                collection_name=mem_cfg["collection_name"],
                vector_size=mem_cfg["vector_size"],
            )
            self.memory = VisualMemory(client=qdrant)
            self.retriever = SceneRetriever(
                client=qdrant,
                top_k=config["retrieval"]["top_k"],
                score_threshold=config["retrieval"]["score_threshold"],
            )
            self.nav = NavigationController(
                confident_threshold=config["retrieval"]["confident_match"],
                partial_threshold=config["retrieval"]["partial_match"],
            )

            self.bridge = CvBridge()

            self.image_sub = self.create_subscription(
                ROSImage, "/camera/image_raw", self._image_callback, 10
            )
            self.decision_pub = self.create_publisher(
                String, "/visual_memory/decision", 10
            )

            self.get_logger().info("VisualMemoryNode initialized")

        def _image_callback(self, msg: ROSImage) -> None:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_image = Image.fromarray(cv_image)

            embedding = self.encoder.encode(pil_image)
            is_kf, emb = self.selector.is_keyframe(embedding)

            if not is_kf:
                return

            payload = MemoryPayload(timestamp=time.time(), room_id=self.room_id)
            self.memory.store(embedding=emb, payload=payload)

            results = self.retriever.query(emb, room_id=self.room_id)
            decision = self.nav.decide(results)

            msg_out = String()
            msg_out.data = decision.model_dump_json()
            self.decision_pub.publish(msg_out)

        def destroy_node(self) -> None:
            self.memory.flush()
            super().destroy_node()

    def main() -> None:
        rclpy.init()
        node = VisualMemoryNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    if __name__ == "__main__":
        main()
