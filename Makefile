NVCC=nvcc
OPENCV_LIBPATH=/usr/local/lib
OPENCV_INCLUDEPATH=/usr/local/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

CUDA_LIBPATH=/usr/local/cuda/lib64
CUDA_INCLUDEPATH=/usr/local/cuda/include
CUDA_LIBS=-lcudart -lcuda

NVCC_OPTS=-O3 -arch=sm_30 -use_fast_math -Xcompiler -Wall

all:
	$(NVCC) -o out  src/*.cpp  src/Benchmark.cu src/Image.cu  src/LPBilinearGpu.cu  src/LPWilsonGpu.cu -L$(OPENCV_LIBPATH) $(OPENCV_LIBS) -I$(OPENCV_INCLUDEPATH) -L$(CUDA_LIBPATH) $(CUDA_LIBS) -I$(CUDA_INCLUDEPATH) $(NVCC_OPTS) 


.PHONY: clean
clean:
	rm -f out %.o  
