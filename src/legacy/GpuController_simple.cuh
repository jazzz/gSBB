#ifndef GPUCONTROLLER_SIMPLE_CUH
#define GPUCONTROLLER_SIMPLE_CUH


#include "defines.h"

#include <cutil_inline.h>
#include <shrQATest.h>

#define WARPSIZE 32

__host__ void InitGPU(int maxLearnerCount , int maxPointCount, int maxProgSize,int pointDim, bool & useGPU);
__host__ void GpuCleanup();

__host__ void EvaluateLearners(int learnerCount, _learner* learnerMatrix, _learnerBid* learnerBidMatrix, int pointCount, _point* pointMatrix );
__host__ void TestLearners(int learnerCount, _learner* learnerMatrix, _learnerBid* learnerBidMatrix, int pointCount, _point* pointMatrix );

__host__ void cLearnerEvalSingle(_learner* lm, _learnerBid &lbm, _point* pm,  int LEARNER_LENGTH, int NUM_FEATURES);
__host__ void RunEvalKernel(int learnersInChunk, int pointsInChunk,int learnerCount, _learner* hLearnerMatrix, _learnerBid* hLearnerBidMatrix, int pointCount, _point* hPointMatrix );

__host__ void resetGPU();






#endif // GPUCONTROLLER_SIMPLE_CUH
#ifdef _DEVICE_VARS_

_point**      dPointPtrArray;
_learner**    dLearnerPtrArray;
_learnerBid* dLearnerBidMatrix;

short numFeatures;
short learnerLength;
short streamCount;

bool isEvalEnvReady;
bool isTestEnvReady;

int bytesize_learner;
int bytesize_singlePoint;
int bytesize_learnerBid;

__constant__ short NUM_FEATURES;
__constant__ short LEARNER_LENGTH;
__constant__ int TOTAL_LEARNERS;
__constant__ int TOTAL_POINTS;

#endif // DEVICE_VARS
