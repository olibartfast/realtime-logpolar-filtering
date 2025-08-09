#pragma once

#include "Image.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace rtlp {

enum class FilterMode {
    BILINEAR,
    BILINEAR_INV,
    BILINEAR_GPU,
    BILINEAR_GPU_INV,
    WILSON,
    WILSON_INV,
    WILSON_GPU,
    WILSON_GPU_INV,
    NONE
};

class VideoProcessor{
	Image *im;
	cv::Mat frame;
	cv::Mat img;
	cv::Mat output;
	time_t start,end, time_last_cycle;
	int fps;
	FilterMode filter_mode;

public:
	VideoProcessor() : im(nullptr), fps(0), filter_mode(FilterMode::NONE) {}
	~VideoProcessor() = default;
	void show();
	void SetImage(Image* i);
	void SetFilter(FilterMode mode);
	void compute_fps(int cnt);

};

} // namespace rtlp
