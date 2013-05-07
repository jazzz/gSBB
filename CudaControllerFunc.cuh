/*
 * CudaController.cuh
 *
 *  Created on: 2013-03-10
 *      Author: jazz
 */

#ifndef CUDACONTROLLERFUNC_CUH_
#define CUDACONTROLLERFUNC_CUH_

#include "defines.h"


//#include <shrQATest.h>

#include <thrust/host_vector.h>
#include <thrust/fill.h>

#define WARPSIZE 32

__host__ void initializeGpuEnv(int maxLearnerCount , int maxPointCount, int maxProgSize,int pointDim);
__host__ void destroyGpuEnv();

__host__ void setConstants(int totalPointCount, int totalLearnerCount, int learnerLength, int numFeatures);

__host__ void allocateDevicePointMatrix();
__host__ void allocateDeviceBidMatrix();

__host__ void freeDevicePointMatrix();
__host__ void freeDeviceBidMatrix();

__host__ void getDevicePointMatrix(_point* &ptr);
__host__ void getDeviceBidMatrix(_learnerBid* &ptr);

__host__ void replacePointInDeviceMatrix(int pointId, _point* hPoint);

__host__ void copyPointMatrixToHost(_point* hPoint);
__host__ void copyBidMatrixToHost(_learnerBid* hBid);

__host__ void pushPointMatrixToDevice(_point* hPoint, int size);
__host__ void pushBidMatrixToDevice(_point* hPoint, int size);


__host__ void startNVTXRange(char* str);
__host__ void stopNVTXRange(char* str);


// Eval Functions
__host__ void EvaluateLearners( _learner* hLearnerMatrix, _learnerBid* dLearnerBidMatrix,_point* dPointMatrix, int learnerCount, int pointCount);
__host__ void cLearnerEvalSingle(_learner* learner, _learnerBid &learnerBid, _point* feature, int LEARNER_LENGTH, int NUM_FEATURES );

// Selection Functions
__host__ void SelectTeamsGPU(int teamCount, int pointCount,_teamReward* rewards, int gapSize,thrust::host_vector<_teamReward>*);
__host__ void SelectPointsGPU(int pointCount, int teamCount,_teamReward* rewards, int gapSize,thrust::host_vector<_teamReward>*);
__global__ void kColumnNormalize(int* dData,int* dBaseSum,int rowCount, int colCount, float* dOut);




#endif /* GPUMEMCONTROLLER_CUH_ */

