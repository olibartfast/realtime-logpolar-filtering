#pragma once

#include "rtlp/core/Image.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

namespace rtlp {
namespace benchmark {

class Benchmark{
	rtlp::core::Image *image_;
	cv::Mat frame;
	cv::Mat img;
	string filename;
	int iterations;
	time_t startCPU, endCPU;

public:
	Benchmark(rtlp::core::Image *img, const std::string& image_path = "test.jpg", int iter = 10)
		: image_(img), filename(image_path), iterations(iter) {}
	~Benchmark() = default;
	void ReadImg();
	void Run();
	void PrintGpuProperties(const cudaDeviceProp& gpuProp); 
	void SaveImg();
};

} // namespace benchmark
} // namespace rtlp
