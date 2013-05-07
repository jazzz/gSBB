#ifndef GPUCONTROLLER_SELECTION_CUH
#define GPUCONTROLLER_SELECTION_CUH


#include "defines.h"
#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/count.h>

#include <cutil_inline.h>
#include <shrQATest.h>

#define WARPSIZE 32

//__host__ void InitializeGPUSelection();


__host__ void SelectTeamsGPU(int teamCount, int pointCount,_teamReward* rewards, int gapSize,thrust::host_vector<int>*);
__host__ void SelectPointsGPU(int pointCount, int teamCount,_teamReward* rewards, int gapSize,thrust::host_vector<int>*);

__global__ void kColumnNormalize(int* dData,int* dBaseSum,int rowCount, int colCount, float* dOut);
//__global__ void kRowSum(float* vec, int vecCount, int rowCount,float* out, int offset);




#endif // GPUCONTROLLER_SIMPLE_CUH
#ifdef _DEVICE_VARS_





#endif // DEVICE_VARS
