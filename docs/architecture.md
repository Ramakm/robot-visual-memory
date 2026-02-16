# Architecture

## System Overview

Robot Visual Memory is a modular system for real-time visual place recognition. It ingests camera frames, selects keyframes, encodes them with CLIP, stores embeddings in Qdrant, and produces navigation decisions.

## Data Flow

```
Camera (30 FPS)
    |
    v
+-------------------+
|  CLIP Encoder     |  src/perception/encoder.py
|  (ViT-B/32)       |
+-------------------+
    |
    v  512-dim embedding
+-------------------+
| Keyframe Selector |  src/perception/keyframe_selector.py
| (cosine distance) |
+-------------------+
    |
    v  keyframe only
+-------------------+     +-------------------+
| Visual Memory     |---->|    Qdrant DB      |
| (batch upsert)    |     | (HNSW + cosine)   |
+-------------------+     +-------------------+
    |                           |
    v                           v
+-------------------+     +-------------------+
| Scene Retriever   |<----|  Filtered Search   |
| (top-k + filter)  |     | (room, timestamp)  |
+-------------------+     +-------------------+
    |
    v  retrieval results
+-------------------+
| Nav Controller    |  src/navigation/controller.py
| (threshold logic) |
+-------------------+
    |
    v
  NavigationDecision
  - LOCALIZE
  - CAUTIOUS_NAVIGATE
  - EXPLORE_AND_MAP
```

## Latency Expectations

| Operation       | Target Latency | Notes                        |
|----------------|---------------|------------------------------|
| CLIP Encode    | 15-30 ms      | GPU (ViT-B/32)              |
| CLIP Encode    | 80-150 ms     | CPU                          |
| Qdrant Insert  | 1-5 ms        | Single point upsert          |
| Qdrant Query   | 2-10 ms       | Top-5, 10k collection        |
| Full Pipeline  | 20-50 ms      | GPU, end-to-end per keyframe |

## Module Dependencies

- `navigation/` depends on `memory/schemas.py` only (no Qdrant, no ROS)
- `retrieval/` depends on `memory/`
- `perception/` is self-contained (CLIP + torch only)
- `ros2/` is optional and isolated
