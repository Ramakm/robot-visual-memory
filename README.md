# robot-visual-memory

Real-time visual place recognition and scene memory for robots using CLIP, Qdrant, and ROS2.

---

## Overview

Robot Visual Memory is a modular system that gives robots the ability to recognize previously visited places in real time. It ingests camera frames, selects keyframes based on visual change, encodes them into CLIP embeddings, stores them in a Qdrant vector database with rich metadata, and produces navigation decisions based on scene similarity.

Key capabilities:

- Ingest camera frames at 30 FPS with intelligent keyframe selection
- Encode frames using CLIP ViT-B/32 into 512-dim embeddings
- Store and retrieve visual memories with metadata filtering (room, timestamp)
- Make navigation decisions: LOCALIZE, CAUTIOUS_NAVIGATE, or EXPLORE_AND_MAP
- Detect scene changes using exponential moving average baselines
- Compress old memories to manage storage growth
- Run standalone or as a ROS2 node

## Architecture

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

## Project Structure

```
robot-visual-memory/
├── config/                  # YAML configuration files
│   ├── default.yaml
│   ├── production.yaml
│   └── benchmark.yaml
├── src/
│   ├── perception/          # CLIP encoder and keyframe selection
│   ├── memory/              # Qdrant client, storage, compression, schemas
│   ├── retrieval/           # Scene retrieval and change detection
│   ├── navigation/          # Navigation decision controller
│   ├── utils/               # Pose, batching, and timing utilities
│   └── main.py              # CLI entry point
├── ros2/                    # Optional ROS2 node and launch file
├── scripts/                 # Standalone CLI tools
├── benchmarks/              # Latency and scaling benchmarks
├── tests/                   # Unit tests (pytest)
└── docs/                    # Architecture and design documentation
```

## Setup

### Prerequisites

- Python 3.10+
- A running Qdrant instance (local or Docker)

### Install Dependencies

```bash
git clone https://github.com/Ramakm/robot-visual-memory.git
cd robot-visual-memory
pip install -r requirements.txt
```

### Start Qdrant

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Docker

Build and run the full stack with Docker Compose:

```bash
docker-compose up --build
```

This starts both Qdrant and the application. Place your input video at `data/input.mp4` and adjust the config as needed.

To run only Qdrant:

```bash
docker-compose up qdrant
```

## Usage

### Process a Video

Run the full pipeline on a video file:

```bash
python -m src.main --video path/to/video.mp4 --room kitchen
```

Options:

```
--video    Path to input video file (required)
--room     Room identifier for metadata (default: "default")
--config   Path to YAML config file (default: config/default.yaml)
```

### Query with an Image

Search visual memory using a single image:

```bash
python scripts/query_cli.py --image path/to/image.jpg --room kitchen --top-k 5
```

### Ingest a Video

Store keyframes from a video without running navigation decisions:

```bash
python scripts/ingest_video.py --video path/to/video.mp4 --room lab
```

### Compress Old Memories

Prune redundant frames from memories older than the configured threshold:

```bash
python scripts/compress_memories.py --config config/default.yaml
```

### Reset Collection

Delete and recreate the Qdrant collection:

```bash
python scripts/reset_collection.py --yes
```

## Benchmarks

Sample results on NVIDIA RTX 3080, Qdrant 1.7 (local Docker):

| Operation       | Mean (ms) | Min (ms) | Max (ms) |
|----------------|----------|---------|---------|
| CLIP Encode    | 18.42    | 16.31   | 24.87   |
| Qdrant Insert  | 2.14     | 1.52    | 5.63    |
| Qdrant Query   | 3.87     | 2.91    | 8.12    |

Scaling behavior (query latency vs collection size):

| Collection Size | Query Latency (ms) |
|----------------|-------------------|
| 1,000          | 2.8               |
| 5,000          | 3.6               |
| 10,000         | 4.2               |

Run benchmarks yourself:

```bash
python benchmarks/latency_profile.py
python benchmarks/performance_test.py
```

## Testing

```bash
pytest tests/ -v
```

Tests use mocked Qdrant clients and do not require a running database.

## Configuration

Configuration is managed through YAML files in `config/`. Key parameters:

| Parameter                        | Default | Description                              |
|----------------------------------|---------|------------------------------------------|
| `keyframe.threshold`             | 0.15    | Cosine distance threshold for keyframes  |
| `retrieval.confident_match`      | 0.85    | Score threshold for LOCALIZE             |
| `retrieval.partial_match`        | 0.75    | Score threshold for CAUTIOUS_NAVIGATE    |
| `memory.collection_name`         | robot_visual_memory | Qdrant collection name        |
| `change_detection.change_threshold` | 0.3  | Delta threshold for scene change         |
| `compression.keep_every_nth`     | 3       | Keep every Nth frame during compression  |

## Roadmap

- REST API for remote querying and memory management
- Multi-floor and outdoor environment support
- Fine-tuned CLIP models for specific robot environments
- Integration with SLAM systems for pose-aware retrieval
- Web dashboard for memory visualization
- Support for additional embedding models (DINOv2, SigLIP)
- Persistent memory export and import

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

Please follow the existing code style: type hints, docstrings, logging instead of print, and separation of concerns.

## License

MIT License. See [LICENSE](LICENSE) for details.
