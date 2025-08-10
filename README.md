# Real-time Log-Polar Filtering

GPU-accelerated implementation of log-polar transformations for real-time image processing.

## Requirements

- OpenCV 4.6+
- CUDA 12.0+
- CMake 3.20+
- C++17 compatible compiler

### Ubuntu/Debian Dependencies

```bash
# Install build tools
sudo apt update
sudo apt install cmake build-essential

# Install OpenCV
sudo apt install libopencv-dev

# For CUDA support, install NVIDIA CUDA Toolkit
# Follow NVIDIA's official installation guide for your system
```

### CMake Options

The following CMake options are available:

- `CMAKE_BUILD_TYPE`: Build type (Release, Debug, RelWithDebInfo)
- `CMAKE_CUDA_ARCHITECTURES`: Target CUDA architectures (auto-detected)
- `OpenCV_DIR`: Custom OpenCV installation path

Example with custom options:
```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DOpenCV_DIR=/custom/opencv/path
```

## Build

### CMake Build (Recommended)

```bash
# Configure the build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build --parallel

# Optional: Install (requires sudo for system-wide installation)
cmake --install build
```

### Alternative Build Method

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Types
```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Release with debug info
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel
```

## Development

### Code Formatting

This project uses clang-format for consistent code formatting. The configuration follows C++ Core Guidelines and modern C++ practices.

To format your code:

```bash
# Format a single file
clang-format -i src/your_file.cpp

# Format all C++ files in the project
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format -i

# Check if files are properly formatted (CI/CD)
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.h" | xargs clang-format --dry-run --Werror
```

### IDE Integration

Most modern IDEs support clang-format integration:
- **VS Code**: Install the "C/C++" extension
- **CLion**: Built-in support, enable in Settings → Editor → Code Style
- **Vim/Neovim**: Use plugins like `vim-clang-format`

## Usage

### Real-time Video Processing
```bash
./rtlp --realtime [FILTER]
```

### Video File Processing
```bash
./rtlp --video [FILTER] --input <input_video> [OPTIONS]
```

**Options:**
- `--input <path>` - Input video file path (required)
- `--output <path>` - Output video file path (default: output.mp4)
- `--frames <n>` - Maximum frames to process (default: all frames)

**Available filters (for both modes):**
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