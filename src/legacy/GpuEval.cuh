/*
 * GpuEval.cuh
 *
 *  Created on: 2013-03-10
 *      Author: jazz
 */

#ifndef GPUEVAL_CUH_
#define GPUEVAL_CUH_

#include "GpuMemController.cuh"

__host__ void EvaluateLearners( _learner* hLearnerMatrix, _learnerBid* dLearnerBidMatrix,_point* dPointMatrix, int learnerCount, int pointCount);
__host__ void cLearnerEvalSingle(_learner* learner, _learnerBid &learnerBid, _point* feature, int LEARNER_LENGTH, int NUM_FEATURES );
#endif /* GPUEVAL_CUH_ */
