# Design Decisions

## Why CLIP?

CLIP (Contrastive Language-Image Pre-training) provides several advantages for visual place recognition:

- **Zero-shot generalization**: CLIP embeddings capture semantic meaning without task-specific fine-tuning, making it robust across diverse indoor and outdoor environments.
- **Compact embeddings**: ViT-B/32 produces 512-dimensional vectors — small enough for real-time operation while retaining high discriminative power.
- **GPU-friendly**: Inference is fast on modern GPUs (15-30ms per frame), enabling real-time operation at keyframe rates.
- **Community support**: OpenCLIP provides open-source weights trained on large-scale datasets (LAION-2B), ensuring reproducibility.

## Why HNSW (via Qdrant)?

Hierarchical Navigable Small World graphs are the indexing strategy used by Qdrant:

- **Sub-linear search**: Query time grows logarithmically with collection size, enabling real-time retrieval even with 100k+ memories.
- **High recall**: HNSW achieves >95% recall at practical speed, which is critical for navigation safety.
- **Dynamic updates**: Unlike tree-based indexes, HNSW supports efficient online insertion — essential for a robot that continuously adds new memories.

## Why Metadata Filtering?

Filtering by `room_id` and `timestamp` before vector search:

- **Reduces search space**: Filtering to a specific room eliminates irrelevant candidates, improving both speed and precision.
- **Temporal coherence**: Timestamp filtering allows the system to focus on recent memories or detect changes over time.
- **Qdrant native support**: Payload indexes enable pre-filtering without post-processing, maintaining low latency.

## Why Keyframe Selection?

Processing every frame at 30 FPS would waste compute on nearly identical consecutive frames:

- **Cosine distance threshold**: Only frames that differ significantly from the last keyframe are processed, typically reducing frame count by 10-20x.
- **Configurable sensitivity**: The threshold can be tuned per environment — tighter for feature-rich spaces, looser for corridors.

## Why Not ROS-Dependent?

The core pipeline is ROS-agnostic by design:

- **Testability**: All modules can be unit-tested without a ROS runtime.
- **Portability**: The system can run on any platform (Docker, cloud, embedded) without ROS installation.
- **ROS2 integration**: An optional ROS2 node wraps the pipeline for robot deployment, subscribing to camera topics and publishing decisions.
