#define EPS2 0.01f
#define G 0.67f

__device__ float3
bodyBodyInteraction2(float4 bi, float4 bj, float3 ai)
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

__global__ void
calculate_forces2(float4 *X, float4 *V, float time)
{
	extern __shared__ float3 acceleration[];
	
	float4 myPosition = X[blockIdx.x];
	float4 myVelocity = V[blockIdx.x];
	float3 acc = { 0.0f, 0.0f, 0.0f };
	
	acceleration[threadIdx.x] = bodyBodyInteraction2(myPosition, X[threadIdx.x], acc);
	__syncthreads();
	
	if(threadIdx.x) return;
	for(int i=0;i<blockDim.x;i++) {
		acc.x += acceleration[i].x;
		acc.y += acceleration[i].y;
		acc.z += acceleration[i].z;
	}
	
	// V = Vo + at
	myVelocity.x += acc.x * time;
    myVelocity.y += acc.y * time;
    myVelocity.z += acc.z * time;
    
    // S = So + Vt
    myPosition.x += myVelocity.x * time;
    myPosition.y += myVelocity.y * time;
    myPosition.z += myVelocity.z * time;
    
    __syncthreads();
    
    X[blockIdx.x] = myPosition;
    V[blockIdx.x] = myVelocity;
    
}

 
void run2(int N, float time, float4 *X, float4 *V) 
{
    int NUM_BLOCKS = N;
    int NUM_THREADS = N;
    int SIZE = N * sizeof(float4);
    
    dim3 block(NUM_THREADS, 1, 1);
    dim3 grid(NUM_BLOCKS, 1, 1);
    
	calculate_forces2<<< grid, block, SIZE>>>(X,V,time);
	cudaDeviceSynchronize();
}
