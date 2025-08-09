#pragma once

#include "Image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace rtlp {

class Benchmark{
	Image *image;
	cv::Mat frame;
	cv::Mat img;
	string filename;
	int iterations;
	time_t startCPU, endCPU;

public:
	Benchmark(Image *img, string image_path = "test.jpg", int iter = 10)
	{
	 image=img;
	 filename=image_path;
	 iterations=iter;
	}
	~Benchmark(){}
	void ReadImg();
	void Run();
	void PrintGpuProperties(const struct cudaDeviceProp gpuProp); 
	void SaveImg();
};

} // namespace rtlp
