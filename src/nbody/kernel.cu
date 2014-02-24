#define EPS2 0.01f
#define G 0.67f

__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
	float3 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;
	
	float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
	float distSixth = distSqr * distSqr * distSqr;
	if(distSixth < 1.0f) return ai;
	float invDistCube = 1.0f/sqrtf(distSixth);
	float s = bj.w * invDistCube;
	
	ai.x += r.x * s * G;
	ai.y += r.y * s * G;
	ai.z += r.z * s * G;
	return ai;
}


__device__ float3
tile_calculation(float4 myPosition, float3 accel)
{
	int i;
	extern __shared__ float4 shPosition[];
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
	}
	return accel;
}


__global__ void
calculate_forces(float4 *X, float4 *V, float time)
{
	extern __shared__ float4 shPosition[];
	
	float4 myPosition = X[threadIdx.x];
	float4 myVelocity = V[threadIdx.x];
	float3 acc = { 0.0f, 0.0f, 0.0f };
	
	shPosition[threadIdx.x] = myPosition;
	__syncthreads();
		
	acc = tile_calculation(myPosition, acc);
	__syncthreads();
	
	
	// V = Vo + at
	myVelocity.x += acc.x * time;
    myVelocity.y += acc.y * time;
    myVelocity.z += acc.z * time;
    
    // S = So + Vt
    myPosition.x += myVelocity.x * time;
    myPosition.y += myVelocity.y * time;
    myPosition.z += myVelocity.z * time;
    
    __syncthreads();
    
    X[threadIdx.x] = myPosition;
    V[threadIdx.x] = myVelocity;
    
}

 
void run(int N, float time, float4 *X, float4 *V) 
{
    int NUM_BLOCKS = 1;
    int NUM_THREADS = N;
    int SIZE = N * sizeof(float4);
    
    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(NUM_BLOCKS, 1, 1);
    
	calculate_forces<<< grid, block, SIZE>>>(X,V,time);
	cudaDeviceSynchronize();
}
