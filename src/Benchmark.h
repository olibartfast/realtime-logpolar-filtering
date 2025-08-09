#pragma once

#include "rtlp/core/Image.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

namespace rtlp {

class Benchmark{
	rtlp::core::Image *image;
	cv::Mat frame;
	cv::Mat img;
	string filename;
	int iterations;
	time_t startCPU, endCPU;

public:
	Benchmark(rtlp::core::Image *img, const std::string& image_path = "test.jpg", int iter = 10)
		: image(img), filename(image_path), iterations(iter) {}
	~Benchmark() = default;
	void ReadImg();
	void Run();
	void PrintGpuProperties(const cudaDeviceProp& gpuProp); 
	void SaveImg();
};

} // namespace rtlp
