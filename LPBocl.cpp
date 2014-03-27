#include "LPBocl.h"

#define LPB_OCL_KERNEL_FILE "LPBoclKernel.cl"


LPBocl::LPBocl(Image *i, bool inv):LogPolar(i,inv){}

LPBocl::~LPBocl()
{
  clReleaseMemObject(xc_d);
  clReleaseMemObject(yc_d);
  clReleaseMemObject(e_d);
  clReleaseMemObject(n_d);

  clReleaseCommandQueue(queue); 
  clReleaseProgram(program); 
  clReleaseContext(context);
}

void LPBocl::process()
{
 create_map();
 to_cortical(); 
 if(inv)
  to_cartesian();
}


void LPBocl::create_map(){
 
  cl_kernel createCorticalMap_kernel, createRetinalMap_kernel;


  //Set platform/device/context
  clGetPlatformIDs(1, &platform, NULL);//In first parameter I assume I have only a device in my system
					//Alternatively I should declare a variable, for instance cl_uint num_platforms,
					//and obtain the number of devices in my system 
// cl_uint num_platforms; 
// clGetPlatformIDs(0, NULL, &num_platforms);
// clGetPlatformIDs(num_platforms, &platform, NULL);
// cl_device_id *cldevs=(cl_device_id*)malloc(num_platforms*sizeof(cl_device_id));
// clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1,cldevs, NULL);
//Context and Command Queue creation
// context = clCreateContext(NULL, 1, cldevs, NULL,NULL, &clerr);
// queue = clCreateCommandQueue(context, cldevs[0], 0, &clerr);
              
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);                               
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &clerr);

  //Read program file
  program_handle = fopen(LPB_OCL_KERNEL_FILE, "r");         
  fseek(program_handle, 0, SEEK_END);//int fseek ( FILE * stream, long int offset, int origin );                         
  program_size = ftell(program_handle); //long int ftell ( FILE * stream ); Get current position in stream              
  rewind(program_handle);                            
  program_buffer = (char*)malloc(program_size + 1);  
  program_buffer[program_size] = '\0';               
  fread(program_buffer, sizeof(char), program_size, program_handle);                                
  fclose(program_handle);

 //Compile program
 program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &program_size, &clerr);          
 free(program_buffer);                                    
 clBuildProgram(program, 0, NULL, NULL, NULL, NULL);  

 //Create  "createCorticalMapKernel" kernel/queue
 createCorticalMap_kernel = clCreateKernel(program, "createCorticalMapKernel", &clerr);      
 queue = clCreateCommandQueue(context, device, 0, &clerr);
 
 //Allocate in GPU table of cartesian values
  xc_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, R*S*sizeof(float), NULL, NULL);
  yc_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, R*S*sizeof(float), NULL, NULL);

  // Set the arguments of the kernel
  clerr = clSetKernelArg(createCorticalMap_kernel, 0, sizeof(float), (void *)&x0);
  clerr = clSetKernelArg(createCorticalMap_kernel, 1, sizeof(float), (void *)&y0);
  clerr = clSetKernelArg(createCorticalMap_kernel, 2, sizeof(float), (void *)&a);
  clerr = clSetKernelArg(createCorticalMap_kernel, 3, sizeof(float), (void *)&q);
  clerr = clSetKernelArg(createCorticalMap_kernel, 4, sizeof(int), (void *)&p0);
  clerr = clSetKernelArg(createCorticalMap_kernel, 5, sizeof(cl_mem), (void *)&xc_d);
  clerr = clSetKernelArg(createCorticalMap_kernel, 6, sizeof(cl_mem), (void *)&yc_d);
  clerr = clSetKernelArg(createCorticalMap_kernel, 7, sizeof(int), (void *)&R);
  clerr = clSetKernelArg(createCorticalMap_kernel, 8, sizeof(int), (void *)&S);

  //Execute kernel	
  size_t local_size [2] = {16,16};
  size_t global_size [2] = {local_size[0]*(R/local_size[0]+1), local_size[1]*(S/local_size[1]+1)}; 
  clEnqueueNDRangeKernel(queue, createCorticalMap_kernel, 2, NULL,global_size, local_size, 0, NULL, NULL); 

  clReleaseKernel(createCorticalMap_kernel); 



if (inv)
 {
  createRetinalMap_kernel = clCreateKernel(program, "createRetinalMapKernel", &clerr);
  //Allocate in GPU table of cartesian values
  e_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, W*H*sizeof(float), NULL, NULL);
  n_d = clCreateBuffer(context, CL_MEM_WRITE_ONLY, W*H*sizeof(float), NULL, NULL);
  // Set the arguments of the kernel
  clerr = clSetKernelArg(createRetinalMap_kernel, 0, sizeof(float), (void *)&x0);
  clerr = clSetKernelArg(createRetinalMap_kernel, 1, sizeof(float), (void *)&y0);
  clerr = clSetKernelArg(createRetinalMap_kernel, 2, sizeof(float), (void *)&a);
  clerr = clSetKernelArg(createRetinalMap_kernel, 3, sizeof(float), (void *)&q);
  clerr = clSetKernelArg(createRetinalMap_kernel, 4, sizeof(int), (void *)&p0);
  clerr = clSetKernelArg(createRetinalMap_kernel, 5, sizeof(cl_mem), (void *)&e_d);
  clerr = clSetKernelArg(createRetinalMap_kernel, 6, sizeof(cl_mem), (void *)&n_d);
  clerr = clSetKernelArg(createRetinalMap_kernel, 7, sizeof(int), (void *)&W);
  clerr = clSetKernelArg(createRetinalMap_kernel, 8, sizeof(int), (void *)&H);
  //Execute kernel	
  size_t local_size2 [2] = {16,16};
  size_t global_size2 [2] = {local_size[0]*(W/local_size[0]+1), local_size[1]*(H/local_size[1]+1)};
  clEnqueueNDRangeKernel(queue, createRetinalMap_kernel, 2, NULL,global_size2, local_size2, 0, NULL, NULL); 
  clReleaseKernel(createRetinalMap_kernel);
 }
}

void LPBocl::to_cortical(){
 int *cort=new int[R*S];

  cl_kernel interp_kernel;
  interp_kernel = clCreateKernel(program, "interpKernel", &clerr);
  // Set the arguments of the kernel
  clerr = clSetKernelArg(interp_kernel, 0, sizeof(cl_mem), (void *)&(imgfilter->GetGpuCPnt()));
  clerr = clSetKernelArg(interp_kernel, 1, sizeof(cl_mem), (void *)&xc_d);
  clerr = clSetKernelArg(interp_kernel, 2, sizeof(cl_mem), (void *)&yc_d);
  clerr = clSetKernelArg(interp_kernel, 3, sizeof(int), (void *)&W);
  clerr = clSetKernelArg(interp_kernel, 4, sizeof(int), (void *)&H);
  clerr = clSetKernelArg(interp_kernel, 5, sizeof(int), (void *)&R);
  clerr = clSetKernelArg(interp_kernel, 6, sizeof(int), (void *)&S);
  clerr = clSetKernelArg(interp_kernel, 7, sizeof(bool), (void *)&true);
  clerr = clSetKernelArg(interp_kernel, 8, sizeof(cl_mem), (void *)&(imgfilter->GetGpuRPnt()));

  //Execute kernel	
  size_t local_size [2] = {16,16};
  size_t global_size [2] = {local_size[0]*(R/local_size[0]+1), local_size[1]*(S/local_size[1]+1)}; 

  clEnqueueNDRangeKernel(queue, interp_kernel, 2, NULL,global_size, local_size, 0, NULL, NULL); 
// interpKernel<<<dimGrid, dimBlock>>>(imgfilter->GetGpuCPnt(), xc_d, yc_d, W, H, R, S,true, imgfilter->GetGpuRPnt());

// cudaMemcpy(cort, imgfilter->GetGpuCPnt(), R*S*sizeof(int), cudaMemcpyDeviceToHost);

  clReleaseKernel(interp_kernel);

 imgfilter->SetData(R,S,cort);
 delete [] cort;
}

void LPBocl::to_cartesian(){
 int *ret= new int [W*H];

//  dim3 dimBlock(16, 16);
//  dim3 dimGrid(W/dimBlock.x+1, H/dimBlock.y+1);
 
// interpKernel<<<dimGrid, dimBlock>>>(imgfilter->GetGpuRPnt(), e_d, n_d, R, S, W, H,false, imgfilter->GetGpuCPnt());

// cudaMemcpy(ret, imgfilter->GetGpuRPnt(), W*H*sizeof(int), cudaMemcpyDeviceToHost);


 imgfilter->SetData(W,H,ret);
 delete [] ret;
}

