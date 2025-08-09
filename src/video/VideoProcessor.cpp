#include "rtlp/video/VideoProcessor.hpp"

#include "rtlp/processing/LPBilinear.hpp"
#include "rtlp/processing/LPBilinearGpu.hpp"
#include "rtlp/processing/LPWilson.hpp"
#include "rtlp/processing/LPWilsonGpu.hpp"

namespace rtlp {
namespace video {

enum class ImageSaveModeLabel {
	LP_BIL,
	LP_BIL_INV,
	LP_BIL_GPU,
	LP_BIL_GPU_INV,
	LP_WIL,
	LP_WIL_INV,
	LP_WIL_GPU,
	LP_WIL_GPU_INV,
	NO_ELAB
};

ImageSaveModeLabel isml;

static string ImageSaveModeTxt[] = { 
	"LP Bilinear",
	"LP Bilinear inverse",
	"LP Bilinear GPU direct",
	"LP Bilinear GPU inverse",
	"LP Wilson",
	"LP Wilson inverse",
	"LP Wilson GPU",
	"LP Wilson GPU inverse",
	"Image not processed"
};



void VideoProcessor::SetImage(rtlp::core::Image* i) {im = i;}

void VideoProcessor::SetFilter(FilterMode mode) {filter_mode = mode;}


void VideoProcessor::show()
{
	cv::VideoCapture capture(0);
	start=clock();
	int counter=0;
	
	cout << "Starting real-time processing with filter: ";
	switch(filter_mode) {
		case FilterMode::BILINEAR:
			cout << "LogPolar direct (Bilinear)" << endl;
			break;
		case FilterMode::BILINEAR_INV:
			cout << "LogPolar direct+inverse (Bilinear)" << endl;
			break;
		case FilterMode::BILINEAR_GPU:
			cout << "LogPolar direct (Bilinear GPU)" << endl;
			break;
		case FilterMode::BILINEAR_GPU_INV:
			cout << "LogPolar direct+inverse (Bilinear GPU)" << endl;
			break;
		case FilterMode::WILSON:
			cout << "LogPolar direct (Wilson)" << endl;
			break;
		case FilterMode::WILSON_INV:
			cout << "LogPolar direct+inverse (Wilson)" << endl;
			break;
		case FilterMode::WILSON_GPU:
			cout << "LogPolar direct (Wilson GPU)" << endl;
			break;
		case FilterMode::WILSON_GPU_INV:
			cout << "LogPolar direct+inverse (Wilson GPU)" << endl;
			break;
		case FilterMode::NONE:
			cout << "No processing (original image)" << endl;
			break;
	}
	cout << "Press 's' to save current frame, 'q' or ESC to exit" << endl;
	
	for (;;)
	{
		auto key = static_cast<char>(cv::waitKey(30));
		capture.read(frame);
		cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		output.create(img.size().height, img.size().width,CV_8UC1);
		im->SetH(img.size().height);
		im->SetW(img.size().width);
		im->SetData(img.size().width, img.size().height);

		for(int row=0;row<img.size().height;row++)
		{
			for(int col=0;col<img.size().width;col++)
			{
				im->GetDataPnt()[row*img.size().width+col]=((unsigned char*)( img.data+row*img.size().width))[col];
				(( output.data+row*img.size().width))[col]=0;

			}	
		}

		im->SetDataGpuR(im->GetDataPnt());
		
		// Check for exit conditions
		if (key == 'q' || key == 'Q' || key == 27) {
			capture.release();
			exit(1);
		}
		
		// Apply the selected filter
		switch(filter_mode){
		case FilterMode::BILINEAR:
				{
					rtlp::processing::LPBilinear  lpbdir(im, false);
					lpbdir.process();
					isml = ImageSaveModeLabel::LP_BIL;
				}
				break;
		
		case FilterMode::BILINEAR_INV:
				{
					rtlp::processing::LPBilinear  lpbinv(im, true);
					lpbinv.process();
					isml = ImageSaveModeLabel::LP_BIL_INV;
				}
				break;
		case FilterMode::BILINEAR_GPU:		
				{
					rtlp::processing::LPBilinearGpu lpbgpudir(im, false);
					lpbgpudir.process();
					isml = ImageSaveModeLabel::LP_BIL_GPU;
				}
				break;
		case FilterMode::BILINEAR_GPU_INV:
				{
					rtlp::processing::LPBilinearGpu lpbgpuinv(im, true);
					lpbgpuinv.process();
					isml = ImageSaveModeLabel::LP_BIL_GPU_INV;
				}
				break;

		case FilterMode::WILSON:
				{
					rtlp::processing::LPWilson  lpwdir(im, false);
					lpwdir.process();
					isml = ImageSaveModeLabel::LP_WIL;
				}
				break;
		case FilterMode::WILSON_INV:
				{
					rtlp::processing::LPWilson  lpwinv(im, true);
					lpwinv.process();
					isml = ImageSaveModeLabel::LP_WIL_INV;
				}
				break;
		case FilterMode::WILSON_GPU:
				{
					rtlp::processing::LPWilsonGpu  lpwgpudir(im, false);
					lpwgpudir.process();
					isml = ImageSaveModeLabel::LP_WIL_GPU;
				}
				break;
		case FilterMode::WILSON_GPU_INV:
				{
					rtlp::processing::LPWilsonGpu  lpwgpuinv(im, true);
					lpwgpuinv.process();
					isml = ImageSaveModeLabel::LP_WIL_GPU_INV;
				}
				break;
		case FilterMode::NONE:
		default:
				isml = ImageSaveModeLabel::NO_ELAB;
				break;
		}
		
		for(int row=0;row<im->GetH();row++)
		{
			for(int col=0;col<im->GetW();col++)
			{
				( output.data+row*output.size().width)[col]=im->GetDataPnt()[row*im->GetW()+col];
			}
		}


	compute_fps(++counter);
	cv::imshow("Cam Feed", output );		

	if( key =='s')
	{
		stringstream save_img;
		time_t now;
		time(&now);
//		tm * ptm=0;
//		struct tm timeinfo;
//		localtime_s(&timeinfo, &now);
//		save_img<<ImageSaveModeTxt[isml]<<ptm->tm_hour<<ptm->tm_min<<ptm->tm_sec<<".jpg";
		save_img << ImageSaveModeTxt[static_cast<int>(isml)] << ".jpg";
		im->WriteData(save_img.str());
	}
	

	}
		
	img.release();
	output.release();
	frame.release();
}


void VideoProcessor::compute_fps(int cnt)
{
	stringstream sstm, sstm2;
	float avg_fps_val, fps_val;
	sstm.precision(2);
	sstm2<<ImageSaveModeTxt[static_cast<int>(isml)];
	cv::putText(output, sstm2.str(), cv::Point(15, 425), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255),1,cv::LINE_AA,false);
	avg_fps_val = cnt*1000.0/difftime(end=clock(),start)*Ttime;
	fps_val = (1000.0/difftime(end=clock(),time_last_cycle))*Ttime;
	sstm<<"avg fps "<<avg_fps_val<<"   fps "<<fps_val;
	cv::putText(output, sstm.str(), cv::Point(15, 450), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255),1,cv::LINE_AA,false);
	time_last_cycle=end;
}

} // namespace video
} // namespace rtlp
