/*
 * CudaController.cuh
 *
 *  Created on: 2013-03-10
 *      Author: jazz
 */

#ifndef CUDACONTROLLERVAR_CUH_
#define CUDACONTROLLERVAR_CUH_

#include "defines.h"


#include <shrQATest.h>

#define WARPSIZE 32

_point* dPointMatrix;
_learnerBid* dLearnerBidMatrix;

int bytesize_learner;
int bytesize_singlePoint;
int bytesize_learnerBid;

int maxLearnerCount;
int maxPointCount;
int learnerLength;
int numFeatures;
#endif
