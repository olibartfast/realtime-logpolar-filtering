#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "Image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


class Benchmark{
	Image *image;
	cv::Mat frame;
	cv::Mat img;
	string filename;
	time_t startCPU, endCPU;

public:
	Benchmark(Image *img)
	{
	 image=img;
	}
	~Benchmark(){}
	void ReadImg();
	void Run();
	void PrintGpuProperties(const struct cudaDeviceProp gpuProp); 
	void SaveImg();
};



#endif
