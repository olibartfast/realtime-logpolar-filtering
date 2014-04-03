#ifndef VIEWER_H
#define VIEWER_H

#include "Image.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


class Viewer{
	Image *im;
	cv::Mat frame;
	cv::Mat img;
	cv::Mat output;
	time_t start,end, time_last_cycle;
	int fps, delay;

public:
	Viewer(){}
	~Viewer(){}
	void show();
	void SetImage(Image* i);
	void compute_fps(int cnt);

};


#endif
