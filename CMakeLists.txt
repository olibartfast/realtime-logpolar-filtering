cmake_minimum_required(VERSION 3.5.1)

project (realtime-logpolar-filtering)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

link_libraries(${OpenCV_LIBS} ${CUDA_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})



if(CUDA_FOUND)
  set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -use_fast_math
)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()


if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()


file( GLOB src_c src/*.cpp )
file( GLOB src_cu src/*.cu )
cuda_compile(benchmark_o src/Benchmark.cu)
cuda_compile(image_o src/Image.cu)
cuda_compile(lpbil_o src/LPBilinearGpu.cu)    
cuda_compile(lpwil_o src/LPWilsonGpu.cu) 
file( GLOB inc src/*.h )



cuda_add_executable(rtlp ${src_c} ${benchmark_o} ${image_o} ${lpbil_o} ${lpwil_o}  )
