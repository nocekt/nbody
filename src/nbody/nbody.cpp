#include "cuda.h"
using namespace std;

CUcontext cuContext;
CUfunction function;
CUresult res;

void cuda_init() {
	cuInit(0);
	CUdevice cuDevice;
    res = cuDeviceGet(&cuDevice, 0);
    res = cuCtxCreate(&cuContext, 0, cuDevice);
	CUmodule cuModule = (CUmodule)0;
    res = cuModuleLoad(&cuModule, "kernel.ptx");
    res = cuModuleGetFunction(&pref1, cuModule, "calculate_forces");
}

void run(int N, float time, float4 *X, float4 *V) {
	
	
    int NUM_BLOCKS = 1;
    int NUM_THREADS = N;
    int SIZE = N * sizeof(float4);
    
    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(NUM_BLOCKS, 1, 1);
    
	calculate_forces<<< grid, block, SIZE>>>(X,V,time);
	cudaDeviceSynchronize();
}

int* radixsort(int *X, int n) {	
	init();
	
	int size = n*sizeof(int);
	
	CUdeviceptr devIn, devOut, temp, temp2;
	cuMemAlloc(&devIn, size);
	cuMemAlloc(&devOut, size);
	cuMemAlloc(&temp, size/1024 + 4);
	res = cuMemAlloc(&temp2, size);
	
	cuMemHostRegister(X,size,0);
	cuMemcpyHtoD(devIn, X ,size);
	
	int blocks = ceil((double)n/1024);
	int k = 0;
	
	void* args1[] = { &devIn, &n, &temp, &temp2, &k };
	void* args2[] = { &temp, &blocks };
	void* args3[] = { &devIn, &devOut, &temp, &temp2, &n, &k, &blocks };

	for(;k<28;k++)
	{
		res = cuLaunchKernel(pref1, ceil(sqrt(blocks)), ceil(sqrt(blocks)), 1, 1024, 1, 1, 0, 0, args1, 0);
		cuCtxSynchronize();
		res = cuLaunchKernel(pref2, 1, 1, 1, 1024, 1, 1, 0, 0, args2, 0);
		cuCtxSynchronize();
		res = cuLaunchKernel(radix, ceil(sqrt(blocks)), ceil(sqrt(blocks)), 1, 1024, 1, 1, 0, 0, args3, 0);
		cuCtxSynchronize();
		swap(devIn,devOut);
	}
	
	cuMemcpyDtoH(X,devIn,size);
	
    cuCtxDestroy(cuContext);
    return X;
}
