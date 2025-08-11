#include "rtlp/benchmark/Benchmark.hpp"
#include "rtlp/processing/LPBilinear.hpp"
#include "rtlp/processing/LPBilinearGpu.hpp"
#include "rtlp/processing/LPWilson.hpp"
#include "rtlp/processing/LPWilsonGpu.hpp"

#include <unistd.h>
#include <fstream>
#include <iomanip>

namespace rtlp {
namespace benchmark {

void Benchmark::ReadImg()
{
	cout << "Loading image: " << filename << endl;
	image_->ReadData(filename);
	image_->SetDataGpuR(image_->GetDataPnt());
	cout << "Image loaded successfully. Size: " << image_->GetW() << "x" << image_->GetH() << endl;
}


void Benchmark::SaveImg()
{
 rtlp::core::Image *tmp=new rtlp::core::Image();
 tmp->SetData(image_->GetW(),image_->GetH(), image_->GetDataPnt());

 image_->WriteData("1_unprocessed.jpg");

 
 //--------------------------------------------------------- 
 rtlp::processing::LPBilinear  lpbdir(image_, false);
 lpbdir.process();
 image_->WriteData("2_1_lpbdir.jpg");

 //--------------------------------------------------------- 
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinear  lpbinv(image_, true);
 lpbinv.process();
 image_->WriteData("2_2_lpbinv.jpg");

 //--------------------------------------------------------- 
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinearGpu lpbgpudir(image_, false);
 lpbgpudir.process();
 image_->WriteData("3_1_lpbgpudir.jpg");

 //--------------------------------------------------------- 
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinearGpu lpbgpuinv(image_, true);
 lpbgpuinv.process();
 image_->WriteData("3_2_lpbgpuinv.jpg");
 
 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilson  lpwdir(image_, false);
 lpwdir.process();
 image_->WriteData("4_1_lpwdir.jpg");
 
 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilson  lpwinv(image_, true);
 lpwinv.process();
 image_->WriteData("4_2_lpwinv.jpg");
 

//---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilsonGpu  lpwgpudir(image_, false);
 lpwgpudir.process();
 image_->WriteData("5_1_lpwgpudir.jpg");
 

 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilsonGpu  lpwgpuinv(image_, true);
 lpwgpuinv.process();
 image_->WriteData("5_2_lpwgpuinv.jpg");

 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
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
 cout<<"Image size: "<<"H: "<<image_->GetH()<<"   W: "<<image_->GetW()<<endl<<endl;

 rtlp::core::Image *tmp=new rtlp::core::Image();
 tmp->SetData(image_->GetW(),image_->GetH(), image_->GetDataPnt());

 N = iterations;
 cout<<"Running benchmark with "<<N<<" iterations..."<<endl<<endl;
 
 // Initialize CSV file
 ofstream csvFile("benchmark_results.csv");
 csvFile << "Iteration,LP_Bilinear_Direct_ms,LP_Bilinear_Inverse_ms,LP_Bilinear_GPU_Direct_ms,LP_Bilinear_GPU_Inverse_ms,";
 csvFile << "LP_Wilson_Direct_ms,LP_Wilson_Inverse_ms,LP_Wilson_GPU_Direct_ms,LP_Wilson_GPU_Inverse_ms" << endl;

 float *avg=new float[8];
 for(int i=0; i<8; i++)
	 avg[i]=0;

 // Individual iteration timings for CSV
 float iterTimes[8];

 for(int i=0; i<N;i++){
 cout<<endl<<"----------------------------------------"<<endl;
 cout<<i+1<<"/"<<N<<endl;
 cout<<"----------------------------------------"<<endl;
 //---------------------------------------------------------



 rtlp::processing::LPBilinear  lpbdir(image_, false);

 startCPU=clock();
 lpbdir.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 iterTimes[0] = time;
 avg[0]=(avg[0]*i+time)/(i+1);
 cout<<"LP Bilinear Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[0]<<" ms"<<endl;
 
 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinear  lpbinv(image_, true);

 startCPU=clock();
 lpbinv.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 iterTimes[1] = time;
 avg[1]=(avg[1]*i+time)/(i+1);
 cout<<"LP Bilinear Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[1]<<" ms"<<endl;
 
 usleep(1000*1000);

 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinearGpu lpbgpudir(image_, false);

 cudaEventRecord(start, 0);
 lpbgpudir.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 iterTimes[2] = time;
 avg[2]=(avg[2]*i+time)/(i+1);
 cout<<endl<<"LP Bilinear GPU Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[2]<<" ms"<<" Speedup: "<<avg[0]/avg[2]<<endl;


 //--------------------------------------------------------- 
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPBilinearGpu lpbgpuinv(image_, true);

 cudaEventRecord(start, 0);
 lpbgpuinv.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 iterTimes[3] = time;
 avg[3]=(avg[3]*i+time)/(i+1);
 cout<<"LP Bilinear GPU Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[3]<<" ms"<<" Speedup: "<<avg[1]/avg[3]<<endl;
 
 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilson  lpwdir(image_, false);

 startCPU=clock();
 lpwdir.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[4] = time;
 avg[4]=(avg[4]*i+time)/(i+1);
 cout<<endl<<"LP Wilson Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[4]<<" ms"<<endl;

 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilson  lpwinv(image_, true);

 startCPU=clock();
 lpwinv.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[5] = time;
 avg[5]=(avg[5]*i+time)/(i+1);
 cout<<"LP Wilson Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[5]<<" ms"<<endl;
 
//---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilsonGpu  lpwgpudir(image_, false);

 startCPU=clock();
 lpwgpudir.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[6] = time;
 avg[6]=(avg[6]*i+time)/(i+1);
 cout<<endl<<"LP Wilson GPU Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[6]<<" ms"<<" Speedup: "<<avg[4]/avg[6]<<endl;

 //---------------------------------------------------------
 image_->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 rtlp::processing::LPWilsonGpu  lpwgpuinv(image_, true);

 startCPU=clock();
 lpwgpuinv.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[7] = time;
 avg[7]=(avg[7]*i+time)/(i+1);
 cout<<"LP Wilson GPU Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[7]<<" ms"<<" Speedup: "<<avg[5]/avg[7]<<endl;

 // Write iteration data to CSV
 csvFile << i+1;
 for(int j=0; j<8; j++) {
     csvFile << "," << fixed << setprecision(3) << iterTimes[j];
 }
 csvFile << endl;
 
 //---------------------------------------------------------
 }




 // Write average times to CSV
csvFile << endl << "AVERAGES";
for(int j=0; j<8; j++) {
    csvFile << "," << fixed << setprecision(3) << avg[j];
}
csvFile << endl;

// Write speedup information
csvFile << endl << "SPEEDUPS,N/A,N/A," << fixed << setprecision(2) 
        << avg[0]/avg[2] << "," << avg[1]/avg[3] << ",N/A,N/A," 
        << avg[4]/avg[6] << "," << avg[5]/avg[7] << endl;

csvFile.close();
cout << endl << "Results saved to benchmark_results.csv" << endl;

cudaDeviceReset();
delete tmp;
delete avg;
}


void Benchmark::PrintGpuProperties(const cudaDeviceProp& gpuProp) {
  cout<<"GPU Name: "<<gpuProp.name<<endl;
  cout<<"Compute Capability: "<<gpuProp.major<<"."<<gpuProp.minor<<endl;

}

} // namespace benchmark
} // namespace rtlp
