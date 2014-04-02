NVCC=nvcc
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
OPENCL_LIBS=-lOpenCL
CUDA_INCLUDEPATH=/usr/include

NVCC_OPTS=-O3 -arch=sm_21 -use_fast_math

imgelab:
	$(NVCC) -o out  LPBocl.cpp LogPolar.cpp  LPBilinear.cpp LPWilson.cpp Viewer.cpp Image.cpp Run.cpp Benchmark.cu Image.cu  LPBilinearGpu.cu  LPWilsonGpu.cu  -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(OPENCL_LIBS) -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS) 


.PHONY: clean
clean:
	rm -f imgelab %.o  
