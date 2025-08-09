#pragma once

#include "rtlp/core/Image.hpp"

#include <chrono>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace rtlp {
namespace video {

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
	rtlp::core::Image *im;
	cv::Mat frame;
	cv::Mat img;
	cv::Mat output;
	time_t start,end, time_last_cycle;
	std::chrono::steady_clock::time_point process_start, process_end;
	int fps;
	FilterMode filter_mode;
	cv::VideoWriter video_writer;
	bool headless_mode;
	double last_process_time_ms;
	std::vector<double> timing_window;
	static const size_t MOVING_AVG_WINDOW = 10;

public:
	VideoProcessor() : im(nullptr), fps(0), filter_mode(FilterMode::NONE), headless_mode(false), last_process_time_ms(0.0) {}
	~VideoProcessor() = default;
	void show();
	void processHeadless(const std::string& input_video, const std::string& output_video, int max_frames = -1);
	void SetImage(rtlp::core::Image* i);
	void SetFilter(FilterMode mode);
	void compute_fps(int cnt);
	void compute_timing_headless(int cnt, cv::Mat& output_frame);

};

} // namespace video
} // namespace rtlp
