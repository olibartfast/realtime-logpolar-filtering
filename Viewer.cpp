#include "Viewer.h"

#include "LPBilinear.h"
#include "LPBilinearGpu.h"
#include "LPWilson.h"
#include "LPWilsonGpu.h"




enum ImageSaveModeLabel {
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

enum ImageSaveModeLabel isml;

static string ImageSaveModeTxt[] = { 
	"LP Bilineare",
	"LP Bilineare inverso",
	"LP Bilineare Gpu dir",
	"LP Bilineare Gpu inverso",
	"LP Wilson",
	"LP Wilson inverso",
	"LP Wilson Gpu",
	"LP Wilson Gpu inverso",
	"Immagine non elaborata"
};



void menu_elab();

void benchmark();


void Viewer::SetImage(Image* i) {im = i;}


void Viewer::show()
{
	
	menu_elab();
	cv::VideoCapture capture(0);
	start=clock();
	int counter=0;
	char filtermode = 0;
	for (;;)
	{
		char key=(char)cv::waitKey(30);
		capture.read(frame);
		cvtColor(frame, img, CV_RGB2GRAY);
		output.create(img.size().height, img.size().width,CV_8UC1);
		im->SetH(img.size().height);
		im->SetW(img.size().width);
		im->SetData(img.size().width, img.size().height);
		if (key!=-1 && key!='s')
			{
				filtermode=key;
				counter=0;
				start=clock();
			}

		for(int row=0;row<img.size().height;row++)
		{
			for(int col=0;col<img.size().width;col++)
			{
				im->GetDataPnt()[row*img.size().width+col]=((unsigned char*)( img.data+row*img.size().width))[col];
				(( output.data+row*img.size().width))[col]=0;

			}	
		}

		im->SetDataGpuR(im->GetDataPnt());
		switch(filtermode){
		case 'q':
		case 'Q':
		case 27:
				capture.release();
			//	cvReleaseCapture(&capture2);
				exit(1);
				break;

		case 'a':
				{
					LPBilinear  lpbdir(im, false);
					lpbdir.process();
					isml=LP_BIL;
				}
				break;
		
		case 'b':
				{
					LPBilinear  lpbinv(im, true);
					lpbinv.process();
					isml=LP_BIL_INV;
				}
				break;
		case 'c':		
				{
					LPBilinearGpu lpbgpudir(im, false);
					lpbgpudir.process();
					isml=LP_BIL_GPU;
				}
				break;
		case 'd':
				{
					LPBilinearGpu lpbgpuinv(im, true);
					lpbgpuinv.process();
					isml=LP_BIL_GPU_INV;
				}
				break;

		case 'i':
				{
					LPWilson  lpwdir(im, false);
					lpwdir.process();
					isml=LP_WIL;
				}
				break;
		case 'j':
			
				{
					LPWilson  lpwinv(im, true);
					lpwinv.process();
					isml=LP_WIL_INV;
				}
				break;
		case 'k':
				{
					LPWilsonGpu  lpwgpudir(im, false);
					lpwgpudir.process();
					isml=LP_WIL_GPU;
				}
				break;
		case 'l':
			
				{
					LPWilsonGpu  lpwgpuinv(im, true);
					lpwgpuinv.process();
					isml=LP_WIL_GPU_INV;
				}
				break;
		default:
				isml=NO_ELAB;
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
		save_img<<ImageSaveModeTxt[isml]<<".jpg";
		im->WriteData(save_img.str());
	}
	

	}
			
	img.release();
	output.release();
	frame.release();
}


void Viewer::compute_fps(int cnt)
{
	stringstream sstm, sstm2;
	float avg_fps_val, fps_val;
	sstm.precision(2);
	sstm2<<ImageSaveModeTxt[isml];
	cv::putText(output, sstm2.str(), cv::Point(15, 425), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255),1,CV_AA,false);
	avg_fps_val = cnt*1000.0/difftime(end=clock(),start)*Ttime;
	fps_val = (1000.0/difftime(end=clock(),time_last_cycle))*Ttime;
	sstm<<"avg fps "<<avg_fps_val<<"   fps "<<fps_val;
	cv::putText(output, sstm.str(), cv::Point(15, 450), cv::FONT_HERSHEY_COMPLEX, 0.75, cv::Scalar(255),1,CV_AA,false);
	time_last_cycle=end;
}

void menu_elab()
{
	system(CLRSCR);

	cout<<"Elabora l'immagine con:"<<endl<<endl;
	cout<<"a:	LogPolar diretto (Bilinerare)"<<endl;
	cout<<"b:	LogPolar diretto+inverso (Bilinerare)"<<endl;
	cout<<"c:	LogPolar diretto (Bilinerare Gpu)"<<endl;
	cout<<"d:	LogPolar diretto+inverso  (Bilinerare Gpu)"<<endl;
	cout<<"i:	LogPolar diretto (Wilson)"<<endl;
	cout<<"j:	LogPolar diretto+inverso  (Wilson)"<<endl;
	cout<<"k:	LogPolar diretto (Wilson Gpu)"<<endl;
	cout<<"l:	LogPolar diretto+inverso (Wilson Gpu)"<<endl;
	cout<<"ESC per uscire"<<endl;
	cout<<"s:	Salva il frame corrente"<<endl<<endl;
	cout<<"Qualsiasi altro tasto per visualizzare l'immagine originaria"<<endl;


}


