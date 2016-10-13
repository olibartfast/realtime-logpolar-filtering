NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

CUDA_LIBPATH=/opt/cuda/lib64
CUDA_INCLUDEPATH=/opt/cuda/include
CUDA_LIBS=-lcudart -lcuda

NVCC_OPTS=-O3 -arch=sm_30 -use_fast_math -Xcompiler -Wall

all:
	$(NVCC) -o out  Run.cpp  LogPolar.cpp LPBilinear.cpp LPWilson.cpp Viewer.cpp Benchmark.cu Image.cu  LPBilinearGpu.cu  LPWilsonGpu.cu  -L$(OPENCV_LIBPATH) $(OPENCV_LIBS) -I$(OPENCV_INCLUDEPATH) -L$(CUDA_LIBPATH) $(CUDA_LIBS) -I$(CUDA_INCLUDEPATH) $(NVCC_OPTS) 


.PHONY: clean
clean:
	rm -f out %.o  
