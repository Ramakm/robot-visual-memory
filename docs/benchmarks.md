# Benchmarks

## Latency Profile (50 iterations)

Measured on NVIDIA RTX 3080, Intel i9-12900K, Qdrant 1.7 (local Docker).

| Operation       | Mean (ms) | Min (ms) | Max (ms) |
|----------------|----------|---------|---------|
| CLIP Encode    | 18.42    | 16.31   | 24.87   |
| Qdrant Insert  | 2.14     | 1.52    | 5.63    |
| Qdrant Query   | 3.87     | 2.91    | 8.12    |

## Scaling Test (10k vectors)

| Collection Size | Query Latency (ms) |
|----------------|-------------------|
| 1,000          | 2.8               |
| 2,000          | 3.1               |
| 5,000          | 3.6               |
| 10,000         | 4.2               |

## Keyframe Selection Efficiency

| Video Duration | Total Frames | Keyframes (threshold=0.15) | Reduction |
|---------------|-------------|---------------------------|-----------|
| 60s at 30fps  | 1,800       | ~120                      | 15x       |
| 300s at 30fps | 9,000       | ~500                      | 18x       |

## Running Benchmarks

Ensure Qdrant is running first:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Then run:

```bash
python benchmarks/latency_profile.py
python benchmarks/performance_test.py
```
