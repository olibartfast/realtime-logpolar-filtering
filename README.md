# Real-time Log-Polar Filtering

GPU-accelerated implementation of log-polar transformations for real-time image processing.

## Requirements

- OpenCV 4.6+
- CUDA 12.0+
- CMake 3.20+
- C++17 compatible compiler

## Build

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Usage

### Real-time Video Processing
```bash
./rtlp --realtime [FILTER]
```

**Available filters:**
- `--bilinear` - Bilinear interpolation (CPU)
- `--bilinear-gpu` - Bilinear interpolation (GPU)
- `--wilson` - Wilson model (CPU)
- `--wilson-gpu` - Wilson model (GPU)
- `--bilinear-inv` - Bilinear with inverse transform
- `--bilinear-gpu-inv` - GPU bilinear with inverse transform
- `--wilson-inv` - Wilson with inverse transform
- `--wilson-gpu-inv` - GPU Wilson with inverse transform
- `--no-filter` - Original image (no processing)

### Benchmark Mode
```bash
./rtlp --benchmark [OPTIONS]
```

**Options:**
- `--image <path>` - Image file path (default: test.jpg)
- `--iterations <n>` - Number of iterations (default: 10)

## Project Structure

```
include/rtlp/
├── core/           # Image class
├── video/          # Real-time video processing
├── processing/     # Log-polar algorithms
├── benchmark/      # Performance benchmarking
└── kernels/        # CUDA kernels

src/
├── core/           # Image implementation
├── video/          # VideoProcessor implementation
├── processing/     # Algorithm implementations
├── benchmark/      # Benchmark implementation
└── kernels/        # CUDA kernel implementations
```

## Algorithms

### Retino-cortical Transformation
![Alt Text](./images/retino-cortical-transformation.png)

### Bilinear Interpolation
Fast log-polar transformation using bilinear interpolation.

![Alt Text](./images/bilinear-interpolation.png)

### Wilson Model
Space-variant transformation based on the Wilson cortical model.

![Alt Text](./images/wilson-model.png)

Both algorithms support:
- Forward transformation (Cartesian → Log-polar)
- Inverse transformation (Log-polar → Cartesian)
- CPU and GPU implementations

## Performance

GPU implementations provide significant speedup over CPU versions:
- Bilinear GPU: ~22-25x speedup
- Wilson GPU: ~42-57x speedup

### Test Configuration
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (Compute Capability 8.6)
- **CPU**: Intel Core i5-11400H @ 2.70GHz (6 cores, 12 threads)
- **OS**: Ubuntu 24.04 
- **Test Image**: 3818x2540 pixels (data/test.jpg)

### Benchmark Results (1 iteration)
| Algorithm | CPU Time (ms) | GPU Time (ms) | Speedup |
|-----------|---------------|---------------|---------|
| Bilinear Direct | 87.57 | 3.97 | 22.05x |
| Bilinear Inverse | 815.47 | 32.29 | 25.26x |
| Wilson Direct | 510.40 | 8.98 | 56.82x |
| Wilson Inverse | 1851.06 | 44.50 | 41.60x |

Results vary based on image size and hardware configuration.