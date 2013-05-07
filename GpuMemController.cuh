///*
// * GpuMemController.cuh
// *
// *  Created on: 2013-03-10
// *      Author: jazz
// */
//
//#ifndef GPUMEMCONTROLLER_CUH_
//#define GPUMEMCONTROLLER_CUH_
//
//#include "defines.h"
//
//#include <cutil_inline.h>
//#include <shrQATest.h>
//
//#define WARPSIZE 32
//
//__host__ void initializeGpuEnv(int maxLearnerCount , int maxPointCount, int maxProgSize,int pointDim);
//__host__ void destroyGpuEnv();
//
//__host__ void allocateDevicePointMatrix();
//__host__ void allocateDeviceBidMatrix();
//
//__host__ void freeDevicePointMatrix();
//__host__ void freeDeviceBidMatrix();
//
//__host__ void getDevicePointMatrix(_point* &ptr);
//__host__ void getDeviceBidMatrix(_learnerBid* &ptr);
//
//__host__ void replacePointInDeviceMatrix(int pointId, _point* hPoint);
//
//__host__ void copyPointMatrixToHost(_point* hPoint);
//
//
//
//_point* dPointMatrix;
//_learnerBid* dLearnerBidMatrix;
//
//int bytesize_learner;
//int bytesize_singlePoint;
//int bytesize_learnerBid;
//
//int maxLearnerCount;
//int maxPointCount;
//int learnerLength;
//int numFeatures;
//#endif /* GPUMEMCONTROLLER_CUH_ */
//
//#ifdef _GPU_MEM_VARS_
//__constant__ short NUM_FEATURES;
//__constant__ short LEARNER_LENGTH;
//__constant__ int TOTAL_LEARNERS;
//__constant__ int TOTAL_POINTS;
//#endif // DEVICE_VARS
