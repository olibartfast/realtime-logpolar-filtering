#pragma once

#include "Image.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace rtlp {

enum FilterMode {
    FILTER_BILINEAR,
    FILTER_BILINEAR_INV,
    FILTER_BILINEAR_GPU,
    FILTER_BILINEAR_GPU_INV,
    FILTER_WILSON,
    FILTER_WILSON_INV,
    FILTER_WILSON_GPU,
    FILTER_WILSON_GPU_INV,
    FILTER_NONE
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
	VideoProcessor(){filter_mode = FILTER_NONE;}
	~VideoProcessor(){}
	void show();
	void SetImage(Image* i);
	void SetFilter(FilterMode mode);
	void compute_fps(int cnt);

};

} // namespace rtlp
