#include "Benchmark.h"
#include "LPBilinear.h"
#include "LPBilinearGpu.h"
#include "LPWilson.h"
#include "LPWilsonGpu.h"






void Benchmark::ReadImg()
{
	system(CLRSCR);
	cout<<"1.Carica immagine da webcam"<<endl;
	cout<<"2.Carica immagine da file"<<endl;
	int c;
	cin>>c;
	switch(c)
	{	
	 case 1:		
		{
		cv::VideoCapture capture(0);
		capture.read(frame);
		cvtColor(frame, img, CV_RGB2GRAY);
		image->SetH(img.size().height);
		image->SetW(img.size().width);
		image->SetData(img.size().width, img.size().height);
		for(int row=0;row<img.size().height;row++)
	 	 for(int col=0;col<img.size().width;col++)
	  	  image->GetDataPnt()[row*img.size().width+col]=((unsigned char*)( img.data+row*img.size().width))[col];
		image->SetDataGpuR(image->GetDataPnt());
		frame.release();
		img.release();
		}
		break;
	case 2:
		{
		system(CLRSCR);	
		cout<<"Inserisci il nome del file"<<endl;
		cin>>filename;
		image->ReadData(filename);
 		image->SetDataGpuR(image->GetDataPnt());
		}
		break;
	}
}


void Benchmark::SaveImg()
{
 Image *tmp=new Image();
 tmp->SetData(image->GetW(),image->GetH(), image->GetDataPnt());

 image->WriteData("1_nonelaborata.jpg");

 
 //--------------------------------------------------------- 
 LPBilinear  lpbdir(image, false);
 lpbdir.process();
 image->WriteData("2_1_lpbdir.jpg");

 //--------------------------------------------------------- 
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinear  lpbinv(image, true);
 lpbinv.process();
 image->WriteData("2_2_lpbinv.jpg");

 //--------------------------------------------------------- 
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpudir(image, false);
 lpbgpudir.process();
 image->WriteData("3_1_lpbgpudir.jpg");

 //--------------------------------------------------------- 
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpuinv(image, true);
 lpbgpuinv.process();
 image->WriteData("3_2_lpbgpuinv.jpg");
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwdir(image, false);
 lpwdir.process();
 image->WriteData("4_1_lpwdir.jpg");
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwinv(image, true);
 lpwinv.process();
 image->WriteData("4_2_lpwinv.jpg");
 
 
//---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpudir(image, false);
 lpwgpudir.process();
 image->WriteData("5_1_lpwgpudir.jpg");
 
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpuinv(image, true);
 lpwgpuinv.process();
 image->WriteData("5_2_lpwgpuinv.jpg");

 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
}

void Benchmark::Run()
{
 
 cudaEvent_t start, stop;	
 cudaDeviceProp gpuProperties;
 float time;
 int N;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaSetDevice(0);
 cudaGetDeviceProperties(&gpuProperties, 0);



 cout<<"/////////////////////////////////////////////////////////"<<endl;
 PrintGpuProperties(gpuProperties);
 cout<<"---------------------------------------------------------"<<endl;
 cout<<"Dimensione immagine: "<<"H: "<<image->GetH()<<"   W: "<<image->GetW()<<endl<<endl;

 Image *tmp=new Image();
 tmp->SetData(image->GetW(),image->GetH(), image->GetDataPnt());

 cout<<"Inserire numero iterazioni"<<endl;
 cin>>N;

 float *avg=new float[8];
 for(int i=0; i<8; i++)
	 avg[i]=0;

 for(int i=0; i<N;i++){
 cout<<endl<<"----------------------------------------"<<endl;
 cout<<i+1<<"/"<<N<<endl;
 cout<<"----------------------------------------"<<endl;
 //---------------------------------------------------------
 LPBilinear  lpbdir(image, false);

 startCPU=clock();
 lpbdir.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 avg[0]=(avg[0]*i+time)/(i+1);
 cout<<"LPBilinear diretto: "<<time<<" ms"<<endl<<" Avg time: "<<avg[0]<<" ms"<<endl;
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinear  lpbinv(image, true);

 startCPU=clock();
 lpbinv.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 avg[1]=(avg[1]*i+time)/(i+1);
 cout<<"LPBilinear diretto+inverso: "<<time<<" ms"<<endl<<" Avg time: "<<avg[1]<<" ms"<<endl;
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpudir(image, false);

 cudaEventRecord(start, 0);
 lpbgpudir.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 avg[2]=(avg[2]*i+time)/(i+1);
 cout<<endl<<"LPBilinear Gpu diretto: "<<time<<" ms"<<endl<<" Avg time: "<<avg[2]<<" ms"<<" Speedup: "<<avg[0]/avg[2]<<endl;


 //--------------------------------------------------------- 
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpuinv(image, true);

 cudaEventRecord(start, 0);
 lpbgpuinv.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 avg[3]=(avg[3]*i+time)/(i+1);
 cout<<"LPBilinear Gpu diretto+inverso: "<<time<<" ms"<<endl<<" Avg time: "<<avg[3]<<" ms"<<" Speedup: "<<avg[1]/avg[3]<<endl;
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwdir(image, false);

 startCPU=clock();
 lpwdir.process();
 endCPU=clock();

 avg[4]=(avg[4]*i+(difftime(endCPU, startCPU)/Ttime))/(i+1);
 cout<<endl<<"LPWilson diretto : "<<difftime(endCPU, startCPU)/Ttime<<"ms"<<endl<<" Avg time: "<<avg[4]<<" ms"<<endl;

 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwinv(image, true);

 startCPU=clock();
 lpwinv.process();
 endCPU=clock();

 avg[5]=(avg[5]*i+(difftime(endCPU, startCPU)/Ttime))/(i+1);
 cout<<"LPWilson diretto+inverso: "<<difftime(endCPU, startCPU)/Ttime<<" ms"<<endl<<" Avg time: "<<avg[5]<<" ms"<<endl;
 
//---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpudir(image, false);

 startCPU=clock();
 lpwgpudir.process();
 endCPU=clock();

 avg[6]=(avg[6]*i+(difftime(endCPU, startCPU)/Ttime))/(i+1);
 cout<<endl<<"LPWilson Gpu : "<<difftime(endCPU, startCPU)/Ttime<<"ms"<<endl<<" Avg time: "<<avg[6]<<" ms"<<" Speedup: "<<avg[4]/avg[6]<<endl;

 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpuinv(image, true);

 startCPU=clock();
 lpwgpuinv.process();
 endCPU=clock();

 avg[7]=(avg[7]*i+(difftime(endCPU, startCPU)/Ttime))/(i+1);
 cout<<"LPWilson GPU diretto+inverso: "<<difftime(endCPU, startCPU)/Ttime<<" ms"<<endl<<" Avg time: "<<avg[7]<<" ms"<<" Speedup: "<<avg[5]/avg[7]<<endl;

 
 //---------------------------------------------------------
 }





 cudaDeviceReset();
 delete tmp;
 delete avg;
}


void Benchmark::PrintGpuProperties(const struct cudaDeviceProp gpuProp) {
  cout<<"Nome della GPU: "<<gpuProp.name<<endl;
  cout<<"Compute Capability: "<<gpuProp.major<<"."<<gpuProp.minor<<endl;

}
