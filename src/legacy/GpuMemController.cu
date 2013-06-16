#include "CudaControllerFunc.cuh"
#include "CudaControllerVars.cuh"

#define _GPU_MEM_VARS_

__host__
void initializeGpuEnv(int _maxLearnerCount , int _maxPointCount, int _maxProgSize,int _pointDim)
{
    numFeatures = _pointDim;
    learnerLength = _maxProgSize+1;

	maxLearnerCount = _maxLearnerCount;
	maxPointCount = _maxPointCount;

    bytesize_learner = sizeof(_learner ) * learnerLength;//maxProgSize;
    bytesize_singlePoint = sizeof(_point) * learnerLength;//maxProgSize;
    bytesize_learnerBid     = sizeof(_learnerBid)  * _maxPointCount;


}
//
//__host__ void destroyGpuEnv()
//{
//
//}
//__host__ void allocateDevicePointMatrix()
//{
//	cutilSafeCall( cudaMalloc( (void**) &dPointMatrix, bytesize_singlePoint*maxPointCount));
//}
//__host__ void allocateDeviceBidMatrix()
//{
//	cutilSafeCall( cudaMalloc( (void**) &dLearnerBidMatrix, bytesize_learnerBid*maxLearnerCount));
//}
//
//__host__ void freeDevicePointMatrix()
//{
//	cutilSafeCall( cudaFree( dPointMatrix));
//}
//__host__ void freeDeviceBidMatrix()
//{
//	cutilSafeCall( cudaFree( dLearnerBidMatrix));
//}
//
//
//
//__host__ void getDevicePointMatrix(_point* &ptr)
//{
//	ptr = dPointMatrix;
//}
//__host__ void getDeviceBidMatrix(_learnerBid* &ptr)
//{
//	ptr = dLearnerBidMatrix;
//}
//
//__host__ void replacePointInDeviceMatrix(int pointId, _point* hPoint)
//{
//	cutilSafeCall(cudaMemcpy(&dPointMatrix[pointId*numFeatures], hPoint, bytesize_singlePoint*maxPointCount, cudaMemcpyDeviceToHost));
//}
//
//__host__ void copyPointMatrixToHost(_point* hPoint)
//{
//	cutilSafeCall(cudaMemcpy( hPoint, dPointMatrix, bytesize_singlePoint*maxPointCount, cudaMemcpyDeviceToHost));
//}
//
//
//
//
//
