#ifndef GPUTESTCONTROLLER_CUH
#define GPUTESTCONTROLLER_CUH


#include "defines.h"

#include <cutil_inline.h>
#include <shrQATest.h>

#define WARPSIZE 32

__host__ void TestLearners(int learnerCount, int learnerLength, _learner* learnerMatrix, _learnerBid* learnerBidMatrix, int pointCount, _point* pointMatrix , int pointDim);



#endif // GPUCONTROLLER_SIMPLE_CUH
#ifdef _DEVICE_VARS_TEST_


__constant__ short NUM_FEATURES;
__constant__ short LEARNER_LENGTH;
__constant__ int TOTAL_LEARNERS;
__constant__ int TOTAL_POINTS;

#endif // DEVICE_VARS
