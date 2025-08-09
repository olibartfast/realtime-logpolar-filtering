#include "Benchmark.h"
#include "LPBilinear.h"
#include "LPBilinearGpu.h"
#include "LPWilson.h"
#include "LPWilsonGpu.h"

#include <unistd.h>
#include <fstream>
#include <iomanip>

namespace rtlp {

void Benchmark::ReadImg()
{
	cout << "Loading image: " << filename << endl;
	image->ReadData(filename);
	image->SetDataGpuR(image->GetDataPnt());
	cout << "Image loaded successfully. Size: " << image->GetW() << "x" << image->GetH() << endl;
}


void Benchmark::SaveImg()
{
 Image *tmp=new Image();
 tmp->SetData(image->GetW(),image->GetH(), image->GetDataPnt());

 image->WriteData("1_unprocessed.jpg");

 
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
 cout<<"Image size: "<<"H: "<<image->GetH()<<"   W: "<<image->GetW()<<endl<<endl;

 Image *tmp=new Image();
 tmp->SetData(image->GetW(),image->GetH(), image->GetDataPnt());

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



 LPBilinear  lpbdir(image, false);

 startCPU=clock();
 lpbdir.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 iterTimes[0] = time;
 avg[0]=(avg[0]*i+time)/(i+1);
 cout<<"LP Bilinear Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[0]<<" ms"<<endl;
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinear  lpbinv(image, true);

 startCPU=clock();
 lpbinv.process();
 endCPU=clock();

 time=difftime(endCPU, startCPU)/Ttime;
 iterTimes[1] = time;
 avg[1]=(avg[1]*i+time)/(i+1);
 cout<<"LP Bilinear Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[1]<<" ms"<<endl;
 
 usleep(1000*1000);

 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpudir(image, false);

 cudaEventRecord(start, 0);
 lpbgpudir.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 iterTimes[2] = time;
 avg[2]=(avg[2]*i+time)/(i+1);
 cout<<endl<<"LP Bilinear GPU Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[2]<<" ms"<<" Speedup: "<<avg[0]/avg[2]<<endl;


 //--------------------------------------------------------- 
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPBilinearGpu lpbgpuinv(image, true);

 cudaEventRecord(start, 0);
 lpbgpuinv.process();
 cudaEventRecord(stop,0);

 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&time, start, stop);

 iterTimes[3] = time;
 avg[3]=(avg[3]*i+time)/(i+1);
 cout<<"LP Bilinear GPU Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[3]<<" ms"<<" Speedup: "<<avg[1]/avg[3]<<endl;
 
 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwdir(image, false);

 startCPU=clock();
 lpwdir.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[4] = time;
 avg[4]=(avg[4]*i+time)/(i+1);
 cout<<endl<<"LP Wilson Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[4]<<" ms"<<endl;

 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilson  lpwinv(image, true);

 startCPU=clock();
 lpwinv.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[5] = time;
 avg[5]=(avg[5]*i+time)/(i+1);
 cout<<"LP Wilson Direct+Inverse: "<<time<<" ms"<<endl<<" Avg time: "<<avg[5]<<" ms"<<endl;
 
//---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpudir(image, false);

 startCPU=clock();
 lpwgpudir.process();
 endCPU=clock();

 time = difftime(endCPU, startCPU)/Ttime;
 iterTimes[6] = time;
 avg[6]=(avg[6]*i+time)/(i+1);
 cout<<endl<<"LP Wilson GPU Direct: "<<time<<" ms"<<endl<<" Avg time: "<<avg[6]<<" ms"<<" Speedup: "<<avg[4]/avg[6]<<endl;

 //---------------------------------------------------------
 image->SetData(tmp->GetW(),tmp->GetH(), tmp->GetDataPnt());
 LPWilsonGpu  lpwgpuinv(image, true);

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

} // namespace rtlp
