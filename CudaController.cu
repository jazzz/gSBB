#define APP_DEBUG


#include "CudaControllerFunc.cuh"
#include "CudaControllerVars.cuh"

#include <thrust/device_vector.h>
#include "ErrorChecking.cuh"
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#include "nvToolsExtMeta.h"

long Diff2(timeval tv_start, timeval tv_end){
   return 1000000*(tv_end.tv_sec - tv_start.tv_sec) + tv_end.tv_usec - tv_start.tv_usec;
}

#define WARP_SIZE 32
#define MODE_MASK 0x1
#define OP_MASK 0xE
#define DST_MASK 0x70
#define SRC_MASK 0x1F80

#define MODE_SHIFT 0
#define OP_SHIFT 1
#define DST_SHIFT 4
#define SRC_SHIFT 7

#define REG_COUNT 8

#define OP_SUM 0
#define OP_DIFF 1
#define OP_PROD 2
#define OP_DIV 3
#define OP_MOD 4
#define OP_COS 5
#define OP_EXP 6
#define OP_LOG 7
#define OP_CODE_COUNT 8


#define OpCodeCount 7
#define OP_NO 199
#define PIVOT_STRIPE_SIZE 16

__constant__ short NUM_FEATURES;
__constant__ short LEARNER_LENGTH;
__constant__ int TOTAL_LEARNERS;
__constant__ int TOTAL_POINTS;

#define KASSERT(condition)  if (condition) ;else{ return; }


typedef texture<int,2,cudaReadModeElementType> tex;
tex texref;
__host__
void initializeGpuEnv(int _maxLearnerCount , int _maxPointCount, int _maxProgSize,int _pointDim)
{
    numFeatures = _pointDim;
    learnerLength = _maxProgSize+1;

	maxLearnerCount = _maxLearnerCount;
	maxPointCount = _maxPointCount;
	DEBUG_PRINT(("InitialVArs: LC:%d   MPC:%d    PS:%d   DIM:%d ",_maxLearnerCount,_maxPointCount,_maxProgSize,_pointDim ));
    bytesize_learner     = sizeof(_learner ) * learnerLength;//maxProgSize;
    bytesize_singlePoint = sizeof(_point) * numFeatures;//maxProgSize;
    bytesize_learnerBid  = sizeof(_learnerBid)  * _maxPointCount;


}
__host__ void setConstants(int totalPointCount, int totalLearnerCount, int learnerLength, int numFeatures)
{
	   CudaSafeCall( cudaMemcpyToSymbol( NUM_FEATURES, &numFeatures,sizeof(short)));
	   CudaSafeCall( cudaMemcpyToSymbol( LEARNER_LENGTH, &learnerLength,sizeof(short)));
	   CudaSafeCall( cudaMemcpyToSymbol( TOTAL_POINTS, &totalPointCount,sizeof(int)));

}


__host__ void destroyGpuEnv()
{

}

__host__ void startNVTXRange(char* str)
{
	nvtxRangePush(str);
//	      nvtxMark("Waiting...");
}
__host__ void stopNVTXRange(char* str)
{
	nvtxRangePop();
}

__host__ void allocateDevicePointMatrix()
{
	CudaSafeCall( cudaMalloc( (void**) &dPointMatrix, bytesize_singlePoint*maxPointCount));
}
__host__ void allocateDeviceBidMatrix()
{
	CudaSafeCall( cudaMalloc( (void**) &dLearnerBidMatrix, bytesize_learnerBid*maxLearnerCount));
}

__host__ void freeDevicePointMatrix()
{
	CudaSafeCall( cudaFree( dPointMatrix));
}
__host__ void freeDeviceBidMatrix()
{
	CudaSafeCall( cudaFree( dLearnerBidMatrix));
}



__host__ void getDevicePointMatrix(_point* &ptr)
{
	ptr = dPointMatrix;
}
__host__ void getDeviceBidMatrix(_learnerBid* &ptr)
{
	ptr = dLearnerBidMatrix;
}

__host__ void replacePointInDeviceMatrix(int pointId, _point* hPoint)
{
	CudaSafeCall(cudaMemcpy(&dPointMatrix[pointId*numFeatures], hPoint, bytesize_singlePoint*maxPointCount, cudaMemcpyDeviceToHost));
}

__host__ void copyPointMatrixToHost(_point* hPoint)
{
	CudaSafeCall(cudaMemcpy( hPoint, dPointMatrix, bytesize_singlePoint*maxPointCount, cudaMemcpyDeviceToHost));
}

__host__ void copyBidMatrixToHost(_learnerBid* hBid)
{
	CudaSafeCall(cudaMemcpy( hBid, dLearnerBidMatrix, bytesize_learnerBid * maxLearnerCount, cudaMemcpyDeviceToHost));
}

__host__ void pushPointMatrixToDevice(_point* hPoint, int size)
{
	_point* hPointPivotMatrix = (_point*) malloc( sizeof(_point) *size*numFeatures);
	int pivotStripeSize = PIVOT_STRIPE_SIZE;

	int numStripes = (size-1)/ pivotStripeSize+1;
	for(int pivotStripeId=0; pivotStripeId < numStripes; pivotStripeId++)
	{
		int offset = pivotStripeId*(pivotStripeSize * numFeatures);
		_point* pointPtr = &hPoint[offset];
		_point* pivotPtr = &hPointPivotMatrix[offset];

		//        for(int i = pivotStripeSize-1; i >0;--i)
		//        {
		//        	for(int j = numFeatures-1;j>0; --j)
		//        	{
		//        		if(i*numFeatures < pointCount)
		//        		pivotPtr[j*pivotStripeSize+i] = 2;// pointPtr[i*numFeatures+j];
		//        	}
		//        }
		for(int i = 0; i < pivotStripeSize; i++)
		{
			for(int j = 0;j < numFeatures; j++)
			{
				//if(i*numFeatures < pointCount)
				pivotPtr[j*pivotStripeSize+i] = pointPtr[i*numFeatures+j];
				//		printf("   %f",pointPtr[i*numFeatures+j] );
			}
			//printf("\n");
		}
	}
	CudaSafeCall(cudaMemcpy(dPointMatrix, hPointPivotMatrix, bytesize_singlePoint * size, cudaMemcpyHostToDevice));
	free(hPointPivotMatrix);
}
__host__ void pushBidMatrixToDevice(_learnerBid* hBid, int size)
{
	CudaSafeCall(cudaMemcpy(dLearnerBidMatrix, hBid, bytesize_learnerBid * size, cudaMemcpyHostToDevice));
}


__global__ void kLearnerEval(int learnOffset,
                      int pointOffset,
                      _learner* dLearnerMatrix,
                      _learnerBid* dLearnerBidMatrix,
                      _point* dPointMatrix,
                      int learnerCount,
                      int pointCount)
{

    int pointId = threadIdx.x + blockIdx.x * blockDim.x + pointOffset; // + (blockIdx.x*gridDim.x);
    int learnerId = threadIdx.y + blockIdx.y* blockDim.y + learnOffset;// + blockIdx.y * blockDim.y;
    if (learnerId < learnerCount && pointId < TOTAL_POINTS)
    {
		int id = threadIdx.x*blockDim.y + threadIdx.y;
		_learner *learner = &dLearnerMatrix[(threadIdx.y + blockIdx.y*blockDim.y) * LEARNER_LENGTH];
        _point *feature = &dPointMatrix[(threadIdx.x+blockDim.x*blockIdx.x)/PIVOT_STRIPE_SIZE * PIVOT_STRIPE_SIZE* NUM_FEATURES + ((threadIdx.x+blockDim.x*blockIdx.x)% PIVOT_STRIPE_SIZE) ];
      //_learner *shared_learner = &dLearnerMatrix[];
        //_point *feature = &dPointMatrix[0];

        KASSERT(LEARNER_LENGTH * blockDim.y < 16*50);
        __shared__ _learner shared_learner[16*50];
        int q=threadIdx.x;
             while(q < LEARNER_LENGTH)
             {
             	shared_learner[threadIdx.y*LEARNER_LENGTH+q] = learner[q];
             	q+= blockDim.x;
             }

        //     __syncthreads();
        _learnerBid registers[8];

        registers[0] =0;
        registers[1] =0;
        registers[2] =0;
        registers[3] =0;
        registers[4] =0;
        registers[5] =0;
        registers[6] =0;
        registers[7] =0;

        if(shared_learner[0] < 1){
     //   dLearnerBidMatrix[ learnerId*TOTAL_POINTS+ pointId] = -1 ;//learnerId+LEARNER_LENGTH ;
        }
        else{
        //short progsize = shared_learner[0];
        	int offset = threadIdx.y*LEARNER_LENGTH;
        for (int i=0;i<=shared_learner[0+offset];i++)
        {
            _learnerBid* dst = &registers[((shared_learner[i+offset] & DST_MASK) >> DST_SHIFT)];

            _learnerBid srcVal;

            if (1 == ((shared_learner[i+offset] & MODE_MASK) >> MODE_SHIFT ) %2) {

               // srcVal =  dPointMatrix[threadIdx.x* NUM_FEATURES + ((shared_learner[i] & SRC_MASK) >> SRC_SHIFT) % NUM_FEATURES ];
                srcVal =  feature[(((shared_learner[i+offset] & SRC_MASK) >> SRC_SHIFT) % NUM_FEATURES)*PIVOT_STRIPE_SIZE ];
            }else{
                srcVal =     registers[(((shared_learner[i+offset] & SRC_MASK) >> SRC_SHIFT) % REG_COUNT)];
            }
            switch ( ((shared_learner[i+offset] & OP_MASK) >> OP_SHIFT) % OP_CODE_COUNT){
            case OP_SUM:
                (*dst) += srcVal;
                break;
            case OP_DIFF:
                (*dst) -= srcVal;
                break;
            case OP_PROD:
                (*dst) *= srcVal;
                break;
            case OP_DIV:
                (*dst) /= srcVal;
                break;
            case OP_MOD:
                (*dst) =  fmod((*dst), srcVal);
                break;
            case OP_COS:
                (*dst) = cos(srcVal);
                break;
            case OP_EXP:
                (*dst) = expf(srcVal);
                break;
            case OP_LOG:
                (*dst) = logf(fabs(srcVal));
                break;
            }
            if(isfinite((*dst)) == 0)
                (*dst) = 0;


        }

       	dLearnerBidMatrix[ learnerId*TOTAL_POINTS + pointId] =   1 / (1+exp(-registers[0]));               //<<----------------------
       // dLearnerBidMatrix[ learnerId*TOTAL_POINTS + pointId] = (threadIdx.y < NUM_FEATURES) ? feature[threadIdx.y ] : -1;
		}
    }
}


__global__ void kTest(_learnerBid* it, _point* p, int learnerCount, int pointCount)
{
	  int pointId = threadIdx.x + blockIdx.x * blockDim.x ;
	  int learnerId = threadIdx.y + blockIdx.y* blockDim.y ;
	  if (learnerId < learnerCount && pointId < pointCount)
	  {

		  it[ learnerId*TOTAL_POINTS + pointId] =  pointId ;
	  }
}
__host__
void EvaluateLearners(_learner* hLearnerMatrix, _learnerBid* bidMatrix,_point* aaadPointMatrix, int learnerCount, int pointCount)
{


	int xThreads = 16;
	int yThreads = 16;

	int xBlocks = (pointCount-1)/xThreads+1;
	int yBlocks = (learnerCount-1)/yThreads+1;


    CudaSafeCall( cudaMemcpyToSymbol( TOTAL_LEARNERS, &learnerCount,sizeof(short)));
    CudaSafeCall( cudaMemcpyToSymbol( TOTAL_POINTS, &pointCount,sizeof(short)));


	_learnerBid* dA;
	CudaSafeCall( cudaMalloc( (void**) &dA, bytesize_learner * learnerCount   ));
	CudaCheckError();
	_learner* dLearnerMatrix;
    CudaSafeCall( cudaMalloc( (void**) &dLearnerMatrix, bytesize_learner * learnerCount   ));
	CudaSafeCall(cudaMemcpy (dLearnerMatrix, hLearnerMatrix, bytesize_learner * learnerCount  , cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemset(dA,1,40));
//	printf("XB:%d  YB:%d  xT:%d  yT:%d  LC:%d  PC:%d\n",xBlocks,yBlocks,xThreads,yThreads,learnerCount,pointCount);
	//kTest<<<dim3(xBlocks,yBlocks),dim3(xThreads,yThreads)>>>(dLearnerBidMatrix,dPointMatrix, learnerCount,pointCount);
	kLearnerEval<<<dim3(xBlocks,yBlocks),dim3(xThreads,yThreads)>>>(0,0,dLearnerMatrix,dLearnerBidMatrix,dPointMatrix,learnerCount, pointCount);
	CudaCheckError();
	//CudaSafeCall(cudaMemcpy (bidMatrix, dA, bytesize_learner * learnerCount   , cudaMemcpyDeviceToHost));
	//copyBidMatrixToHost(bidMatrix);
//	_learnerBid A[5];
//	A[0] = 5;
//	A[1] = 5;
//	A[2] = 5;
//	A[3] = 5;
//	A[4] = 5;
	//aCudaSafeCall(cudaMemcpy (dLearnerBidMatrix, &A, 5  , cudaMemcpyHostToDevice));

	//bidMatrix[5] = 2;
	cudaFree(dLearnerMatrix);
	cudaFree(dA);
}



__host__ void
cLearnerEvalSingle(_learner* learner, _learnerBid &learnerBid, _point* feature, int LEARNER_LENGTH, int NUM_FEATURES )
{

    // printf("cLEArn SINGLE\n");


    float testVal =-1;

    _learnerBid registers[REG_COUNT] ={0};

//		printf("FEATURE ");
//	for(int _a=0;_a< NUM_FEATURES;_a++)
//	{
//		printf(" %f",feature[_a]);
//	}
//		printf("\n");

    short progsize = learner[0];
//     printf(" PROGSIZE: %d" , progsize);
    short val = 0;
    for (int i=1;i<progsize+1;i++)
    {
        unsigned short instruction = learner[i] ;//& 8191;
              // printf( " INST:%d ", instruction);

//		unsigned short inst = instruction;
//		char str[14];
//		str[13] = '\0';
//		for(int p =0 ; p < 13; p++)
//		{
//			str[12-p] = (inst%2==0) ? '0':'1';
//			inst /=2;
//		}
//		printf("%s",str);


        unsigned char mode =0;
        mode = ((instruction & MODE_MASK) >> MODE_SHIFT ) %2;
        char op = 0;
        op = ((instruction & OP_MASK) >> OP_SHIFT) % OP_CODE_COUNT;
        short dst_index = (instruction & DST_MASK) >> DST_SHIFT ; //% REG_COUNT ;
//          printf(" R[%d] <- R[%d] ", dst_index, dst_index);
        _learnerBid* dst = &registers[(instruction & DST_MASK) >> DST_SHIFT];
        unsigned short src = (instruction & SRC_MASK) >> SRC_SHIFT;
        //  float srcVal =	(mode == 1) ? feature[src%NUM_FEATURES] : registers[src%REG_COUNT]   ;
        _learnerBid srcVal;
        // printf(" Mode:%d R[%d]",mode, dst_index );


        if ( mode ==1) {
            srcVal = feature[src%NUM_FEATURES];
//               printf("I[%d] ", src%NUM_FEATURES);
        }else{
            srcVal =     registers[src%REG_COUNT];
//              printf("R[%d] ", src%REG_COUNT);
        }

        //printf(" OP:%d (", op);
        switch (op){
        case OP_SUM:
            (*dst) += srcVal;
//             printf("sum ");
            testVal=srcVal;
            break;
        case OP_DIFF:
            (*dst) -= srcVal;
//                 printf("diff ");
            testVal=1;
            break;
        case OP_PROD:
            (*dst) *= srcVal;
//               printf("prod ");
            testVal=2;
            break;
        case OP_DIV:
            (*dst) /= srcVal;
//               printf("div ");
            testVal=3;
            break;
        case OP_MOD:
            (*dst) =  fmod((*dst), srcVal);
//              printf("mod ");
            testVal=4;
            break;
        case OP_COS:
            (*dst) = cos(srcVal);
//                    printf("cos ");
            break;
        case OP_EXP:
        	//if(srcVal > 88 && srcVal < 709.783){(*dst)=FLT_MAX;}										// Floats Wrapping with EXP -- easiest way to stop it
        	(*dst) = exp(srcVal);
//                   printf("exp ");

            break;
        case OP_LOG:
            (*dst) = logf(fabs(srcVal));
//                printf("log ");
            break;
        }
        if(isfinite((*dst)) == 0)
        {
//        	if( isinf((*dst)) )
//        	{
//        		(*dst) = FLT_MAX;
//        	}
//        	//printf(" DSTNOTFINITE: %f", (*dst) );
//        	else{
            (*dst) = 0;
 //       	}
        }
//                printf("REG ");
//                for ( int i = 0; i < REG_COUNT; i++)
//                {
//
//                    printf("%f ", registers[i]);
//                }
//                printf("\n");
    }
    //        printf(" final:: ");
    //      for ( int i = 0; i < REG_COUNT; i++)
    //      {
    //          printf("   %f ", registers[i]);
    //      }

//    printf(":%f", 1 / float(1+exp(-registers[0])));
    learnerBid = 1 / float(1+exp(-registers[0]));
}



#define BIGNUMBER 99999;
#define STATE_FRONT_TOO_SMALL 0
#define STATE_FRONT_TOO_BIG 1
#define STATE_FRONT_IS_JUUUST_RIGHT 2

int MSB(int v)
{
const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
const unsigned int S[] = {1, 2, 4, 8, 16};
int i;

register unsigned int r = 0; // result of log2(v) will go here
for (i = 4; i >= 0; i--) // unroll for speed...
{
  if (v & b[i])
  {
    v >>= S[i];
    r |= S[i];
  }
}
return r;
}

int nextPowerOf2(int v)
{
	return 1<<(MSB(v)+1);
}


__global__ void kCalcDist(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
	if(threadIdx.x < teamCount && blockIdx.x < pointCount)
	{
		_teamReward* dist;
	//	_teamReward* rewards;
		int j = threadIdx.x;

		dist = dist_out +  blockIdx.x*teamCount*teamCount;

		for(int i=0;i < teamCount; i++)
		{
				dist[j+i*pointCount] = (rewards[i*teamCount+blockIdx.x] > rewards[j*teamCount+blockIdx.x]);

			//	if(j==1){dist[j+i*teamCount] = 1;}
		}

	}
}
// lets assume 16x16 for fun
__global__ void kCalcDistSoMuchBetterBro__(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
	int pointIndex = blockIdx.z;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

//	extern __shared__ _teamReward XCache[];					GAH You 2 hours debug later: TL;DR can't have two separate arrays
//	extern __shared__ _teamReward YCache[];
	extern __shared__ _teamReward Cache[];
	if(i < teamCount && j < teamCount && pointIndex < pointCount)
	{

			if(0 == threadIdx.y )
			{
				Cache[threadIdx.x] = rewards[j*pointCount+pointIndex];
			}
			if(0 == threadIdx.x )
			{
				Cache[threadIdx.y+blockDim.x] = rewards[i*pointCount+pointIndex];
			}

	}
		__syncthreads();
		if(i < teamCount && j < teamCount && pointIndex < pointCount)
		{
			dist_out[pointIndex*teamCount*teamCount + i*teamCount+j] =  Cache[threadIdx.y+blockDim.x] > Cache[threadIdx.x];
		}





}

// lets assume 16x16 for fun
__global__ void kCalcDistSoMuchBetterBro(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
//	int pointIndex = blockIdx.z;
//	int i = blockIdx.y*blockDim.y + threadIdx.y;
//	int j = blockIdx.x*blockDim.x + threadIdx.x;

		if(threadIdx.x < teamCount && threadIdx.y < teamCount)
		{

			dist_out[threadIdx.y*blockDim.x+threadIdx.x] = rewards[threadIdx.y+pointCount*blockIdx.z] > rewards[threadIdx.x+pointCount*blockIdx.z];
//			dist_out[pointIndex*teamCount*teamCount + j*teamCount+i] = 3;
//			dist_out[pointIndex*teamCount*teamCount + j] = 3;
		}
//




}

__global__ void kCalcDistTex(_teamReward* dist_out, int teamCount, int pointCount)
{
	int pointIndex = blockIdx.z;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

		if(i < teamCount && j < teamCount && pointIndex < pointCount)
		{

			dist_out[pointIndex*teamCount*teamCount + i*teamCount+j] = tex2D(texref,j,pointIndex) < tex2D(texref,i,pointIndex) ;
		}
//




}

__host__ void GetDist(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
	//int threads = 16;
	//int xThreads = threads;
	//int yThreads = threads;
	int xThreads = 16;
	int yThreads = 16;


	int xBlocks = (teamCount-1) / xThreads +1;
	int yBlocks = (teamCount-1) / yThreads +1;
	int zBlocks = pointCount;


	//cudaMemset(dist_out,0,sizeof(_teamReward) * teamCount*teamCount*pointCount);


	if(xThreads*yThreads > 1024){ fprintf(stderr,"Error: Too many threads used in GetDist");}
	//kCalcDistSoMuchBetterBro<<<dim3(xBlocks,yBlocks,zBlocks),dim3(xThreads),xThreads>>>(rewards, dist_out, teamCount, pointCount);
	kCalcDistSoMuchBetterBro__<<<dim3(xBlocks,yBlocks,zBlocks),dim3(xThreads,yThreads),xThreads+yThreads>>>(rewards, dist_out, teamCount, pointCount);
	CudaCheckError();

}

__host__ void GetDistTex(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
	//int threads = 16;
	//int xThreads = threads;
	//int yThreads = threads;
	int xThreads = 16;
	int yThreads = 16;


	int xBlocks = (teamCount-1) / xThreads +1;
	int yBlocks = (teamCount-1) / yThreads +1;
	int zBlocks = pointCount;


	//cudaMemset(dist_out,0,sizeof(_teamReward) * teamCount*teamCount*pointCount);

	cudaArray* carray;
	cudaChannelFormatDesc channel;

	channel = cudaCreateChannelDesc<int>();

	int rowCount = teamCount;
	int colCount = pointCount;
	cudaMallocArray(&carray,&channel,rowCount, colCount);
	cudaMemcpyToArray(carray,0,0,rewards,sizeof(int)* rowCount*colCount, cudaMemcpyHostToDevice);

	texref.filterMode=cudaFilterModePoint;
	texref.addressMode[0]=cudaAddressModeClamp;
	texref.addressMode[1]=cudaAddressModeClamp;
	cudaBindTextureToArray(texref,carray);
	if(xThreads*yThreads > 1024){ fprintf(stderr,"Error: Too many threads used in GetDist");}
	kCalcDistTex<<<dim3(xBlocks,yBlocks,zBlocks),dim3(xThreads,yThreads),xThreads+yThreads>>>(dist_out, teamCount, pointCount);
	CudaCheckError();

	cudaUnbindTexture(texref);
	cudaFree(carray);

}


__global__ void isDominated___(_teamReward* Vec, int size, int vecCount,bool* DOM, bool* EQUAL, int* OUT)
{

	int isIndex = blockIdx.x;
	int byIndex = blockIdx.y;


		_teamReward* vIs= &Vec[size*isIndex];
		_teamReward* vBy= &Vec[size*byIndex];
		__shared__ bool dominated[128];
		__shared__ bool equal[128];

		__shared__ bool isDominated;
		__shared__ bool isEqual;
		if(threadIdx.x ==0){isDominated = true; isEqual =true;}
		__syncthreads();

		int elementOffset = 0;
		while(isDominated && elementOffset*blockDim.x < size)
		{
			int tid = 0;
			tid = threadIdx.x + elementOffset;


			//int active_threads = (size - stepIndex*blockDim.x > blockDim.x) ? blockDim.x : size ;
			if(tid < size)
			{
				dominated[threadIdx.x] = (vIs[tid] > vBy[tid]);
				equal[threadIdx.x] = (vIs[tid] == vBy[tid]);
			}else{
				dominated[threadIdx.x] = 0;
				equal[threadIdx.x] = 1;
			}

			__syncthreads();

			int activeThreads = blockDim.x/2;

			while(activeThreads >0)
			{
				if(threadIdx.x < activeThreads && tid+activeThreads < size)
				{
					dominated[threadIdx.x] = dominated[threadIdx.x] || dominated[threadIdx.x+activeThreads];
					equal[threadIdx.x] = equal[threadIdx.x] && equal[threadIdx.x+activeThreads];
				}

				__syncthreads();
				activeThreads /=2;
			}

		__syncthreads();
			if(threadIdx.x ==0)
			{
				isDominated = !dominated[0];
				isEqual = isEqual && equal[0];
					//DOM[isIndex*blockDim.y + byIndex] = dominates[0];
					//EQUAL[isIndex*blockDim.y + byIndex] = equal[0];
			}

			elementOffset += blockDim.x;



		}

		__syncthreads();

		if(threadIdx.x ==0)
		{

			//EQUAL[isIndex*vecCount + byIndex] = isEqual;
			DOM[isIndex*vecCount + byIndex] = (byIndex < isIndex && isEqual) || (isDominated && !isEqual);
			OUT[isIndex*vecCount + byIndex] =  dominated[0];
		}



}
__global__ void isDominated_single(_teamReward* Vec, int size, int vecCount,bool* DOM, bool* EQUAL, int* OUT)
{

	int isIndex = blockIdx.x;
	int byIndex = blockIdx.y;
	int offset = blockIdx.z * threadIdx.x;


	_teamReward* vIs= &Vec[size*isIndex];
	_teamReward* vBy= &Vec[size*byIndex];
	__shared__ bool dominated[128];
	__shared__ bool equal[128];

}


__global__ void isDominated(_teamReward* Vec, int size, int vecCount,bool* DOM, bool* EQUAL, int* OUT)
{

	int isIndex = blockIdx.x;
	int byIndex = blockIdx.y;


		_teamReward* vIs= &Vec[size*isIndex];
		_teamReward* vBy= &Vec[size*byIndex];
		__shared__ bool dominated[128];
		__shared__ bool equal[128];

		__shared__ bool isDominated;
		__shared__ bool isEqual;
		if(threadIdx.x ==0){isDominated = true; isEqual =true;}
		__syncthreads();

		int elementOffset = 0;
		while(isDominated && elementOffset*blockDim.x < size)
		{
			int tid = 0;
			tid = threadIdx.x + elementOffset;


			//int active_threads = (size - stepIndex*blockDim.x > blockDim.x) ? blockDim.x : size ;
			if(tid < size)
			{
				dominated[threadIdx.x] = (vIs[tid] > vBy[tid]);
				equal[threadIdx.x] = (vIs[tid] == vBy[tid]);
			}else{
				dominated[threadIdx.x] = 0;
				equal[threadIdx.x] = 1;
			}

			__syncthreads();

			int activeThreads = blockDim.x/2;

			while(activeThreads >0)
			{
				if(threadIdx.x < activeThreads && tid+activeThreads < size)
				{
					dominated[threadIdx.x] = dominated[threadIdx.x] || dominated[threadIdx.x+activeThreads];
					equal[threadIdx.x] = equal[threadIdx.x] && equal[threadIdx.x+activeThreads];
				}

				__syncthreads();
				activeThreads /=2;
			}

		__syncthreads();
			if(threadIdx.x ==0)
			{
				isDominated = !dominated[0];
				isEqual = isEqual && equal[0];
					//DOM[isIndex*blockDim.y + byIndex] = dominates[0];
					//EQUAL[isIndex*blockDim.y + byIndex] = equal[0];
			}

			elementOffset += blockDim.x;



		}

		__syncthreads();

		if(threadIdx.x ==0)
		{

			//EQUAL[isIndex*vecCount + byIndex] = isEqual;
			DOM[isIndex*vecCount + byIndex] = (byIndex < isIndex && isEqual) || (isDominated && !isEqual);
			OUT[isIndex*vecCount + byIndex] =  isDominated;
		}



}


// TODO: BRO fix  this -- Gonn ahve to loop over an do a recursive reduction
__global__ void findParetoFront(bool* DOM, bool* front, int size)
{
	int pointId = blockIdx.y + blockIdx.x*gridDim.x;
	int index = threadIdx.x;


	extern __shared__ bool data[];



	int offset = blockDim.x;
	data[threadIdx.x] = ((index < size) ? DOM[pointId*size+index]: 0) or ((index + offset < size) ? DOM[pointId*size + index + offset] : 0);



	// Reduce
	__syncthreads();
	int activeThreads = blockDim.x/2;
	while(activeThreads > 0)
	{
		if(threadIdx.x < activeThreads)
			data[threadIdx.x] = data[threadIdx.x] or data[threadIdx.x + activeThreads];
		__syncthreads();
		activeThreads /= 2;
	}

	__syncthreads();
	if(0 == threadIdx.x )
		front[pointId] = data[0];


}

__global__ void findParetoFront2(bool* DOM, bool* out,int size, int pointOffset, int* out2)
{
	int rowId = blockIdx.y ;//+ pointOffset;
	int index = threadIdx.x ;//+ blockIdx.x*blockDim.x;


	extern __shared__ bool data[];


	int offset = blockDim.x;
	data[threadIdx.x] = ((index < size) ? DOM[rowId*size+index]: 0) or ((index + offset < size) ? DOM[rowId*size + index + offset] : 0);


	// Reduce
	__syncthreads();
	int activeThreads = blockDim.x/2;
	while(activeThreads > 0)
	{
		if(threadIdx.x < activeThreads)
			data[threadIdx.x] = data[threadIdx.x] or data[threadIdx.x + activeThreads];
		__syncthreads();
		activeThreads /= 2;
	}

	__syncthreads();
	if(0 == threadIdx.x )
	{
		out[rowId*gridDim.x+blockIdx.x] = !data[0];
	}



}
__global__ void col2Row(bool* to, bool* from,int rowCount,int colCount)
{
	int a;
	if( threadIdx.x < rowCount)
	{
		to[threadIdx.x] = from[threadIdx.x*colCount];
	}

}

__host__ void FindParetoFront(bool* dom, bool* front,int size)	// PS its  rowSize * rowSize Matrix
{
	int rowsPerInvocation = size;

	int colCount = size;
	int rowCount = size;

	int xThreads = 128;
	int xBlocks = (colCount-1) / xThreads +1;
	int yBlocks = rowCount;


	CudaCheckError();
//	bool* A = new bool[rowCount*colCount];
//	CudaSafeCall(cudaMemcpy (A, dom, sizeof(bool) * colCount * rowCount, cudaMemcpyDeviceToHost));
//
//	printf("====^^^^^====\n");
//		for(int i =0; i < rowCount; i++)
//		{
//			for(int j=0;j<colCount;j++)
//			{
//				printf(" %s" , (A[i*colCount+j]) ? "1":"0" );
//			}
//			printf("\n");
//		}

		int colsLeft = colCount;



	bool* dStaging;
	CudaSafeCall( cudaMalloc( (void**) &dStaging,sizeof(bool) * rowCount * xBlocks   ));
	int* tmp;
		CudaSafeCall( cudaMalloc( (void**) &tmp,sizeof(int) * rowCount * rowCount   ));
	//	CudaSafeCall(cudaMemcpy (dWorking, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));

	int rowOffset = 0;

						// TODO: Which way? X blocks First or Y Block First (X = finnish Summation? Y =Finish all rows first
	while(colsLeft > 1)
	{
		xBlocks = (colsLeft-1)/xThreads+1;

		while(rowOffset < yBlocks)
		{

	//		printf("BSS %d\n", xBlocks);
			int blocksPerGrid_x =  xBlocks;
			int blocksPerGrid_y = (rowCount > rowsPerInvocation) ? rowsPerInvocation : rowCount;

			findParetoFront2<<<dim3(1,blocksPerGrid_y),xThreads,xThreads>>>(dom,dStaging,colsLeft,rowOffset,tmp);
			CudaCheckError();
			rowOffset += blocksPerGrid_y;
		}
		colsLeft = xBlocks;
	}
		int t[rowCount*rowCount];
//		CudaSafeCall(cudaMemcpy (t, tmp, sizeof(int) * rowCount *rowCount, cudaMemcpyDeviceToHost));
//		CudaSafeCall(cudaMemcpy (A, dStaging, sizeof(bool) * xBlocks *rowCount, cudaMemcpyDeviceToHost));
//		printf("====TMP====");
//		for(int i =0; i < rowCount; i++)
//				{
//					for(int j=0;j< rowCount;j++)
//					{
//						printf(" %d" , t[i*rowCount+j]);
//					}
//					printf("\n");
//				}
//		printf("====#====");
//		for(int i =0; i < rowCount; i++)
//		{
//			for(int j=0;j< xBlocks;j++)
//			{
//				printf(" %s" , (A[i*xBlocks+j]) ? "1":"0" );
//			}
//			printf("\n");
//		}
//
//	printf(" R == %d\n" ,nextPowerOf2(1204));
	col2Row<<<1,nextPowerOf2(rowCount)>>>(front,dStaging,rowCount,xBlocks);					// LIMIT: rowCount == 1024
	CudaCheckError();

	//	CudaSafeCall(cudaMemcpy (front, dStaging, sizeof(bool) *rowCount, cudaMemcpyDeviceToDevice));

//	CudaSafeCall(cudaMemcpy (A, front, sizeof(bool) * rowCount, cudaMemcpyDeviceToHost));
//		printf("========");
//		for(int i =0; i < rowCount; i++)
//		{
////			for(int j=0;j< xBlocks;j++)
////			{
//				printf(" %s" , (A[i]) ? "1":"0" );
////			}
//			printf("\n");
//		}
//
//

//	delete A;
	cudaFree(tmp);
	cudaFree(dStaging);


}
__global__ void kColumnSum(int* vec, int vecCount, int rowCount,int* out, int offset)
{
	int vecIndex = blockIdx.x + offset;
	int rowIndex = threadIdx.y + blockIdx.y*blockDim.y;
	__shared__ int sum_vec[256];

	sum_vec[threadIdx.y] = 0;

	if(rowIndex < rowCount)
	{
		sum_vec[threadIdx.y] = vec[vecIndex+ rowIndex*vecCount]; //vec[vecIndex + pointCount*vecCount ];
	}



int i = blockDim.y;
i /=2 ;
while(i>0)
{
	__syncthreads();
	if( threadIdx.y < i  )
	{
		sum_vec[threadIdx.y] = sum_vec[threadIdx.y]+ sum_vec[threadIdx.y+ i];//threadIdx.y* 1000 + threadIdx.y+(i/2);

	}
	i /=2 ;

}


	__syncthreads();

	if(threadIdx.y==0)
	{
		out[vecIndex + blockIdx.y*vecCount] = sum_vec[0];

	}


}

__host__ void ColumnSum(int* vec, int rowCount, int colCount)
{
	int yThreads = 128;
	int yBlocks = (rowCount -1)/ yThreads +1;
	CudaCheckError();
	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;
	for(int _rowsLeft=rowCount; _rowsLeft >1;_rowsLeft = (_rowsLeft -1) / yThreads+1)
	{
		int elementsRemaining = colCount;
		for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
		{
			kColumnSum <<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(vec,colCount, rowCount, vec,offset);
			CudaCheckError();
		}

	}
}

template< typename T >
__global__ void kRowSum(T* vec, int rowCount, int vecCount,T* out, int offset)
{
	int vecIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int rowIndex = blockIdx.y + offset;
	__shared__ T sum_vec[256];

	sum_vec[threadIdx.x] = 0;

//	if(rowIndex < rowCount)
//	{
//		sum_vec[threadIdx.x] = vec[vecIndex+ rowIndex*vecCount] ; //vec[vecIndex + pointCount*vecCount ];
//	}
	sum_vec[threadIdx.x] = ((vecIndex < vecCount) ?  vec[vecIndex+ rowIndex*vecCount]: 0); //or ((index + offset < size) ? DOM[pointId*size + index + offset] : 0);


	int i = blockDim.x/2;
	while(i > 0)
	{
		__syncthreads();
		if(threadIdx.x <i)
			sum_vec[threadIdx.x] = sum_vec[threadIdx.x]+ sum_vec[threadIdx.x+ i];//threadIdx.y* 1000 + threadIdx.y+(i/2);
		i /=2;
	}



	__syncthreads();

	if(threadIdx.x  < vecCount)
	{
		out[vecIndex + rowIndex*vecCount] = sum_vec[threadIdx.x];

	}


}
__host__ void RowSum(float* dVec, int rowCount, int colCount)
{
	int xThreads = 128;
	int xBlocks =  (colCount -1)/ xThreads +1;
	int blocksPerGrid_y = (rowCount > 40) ? 40 : rowCount;
	for(int _rowsLeft=colCount; _rowsLeft >1;_rowsLeft = (_rowsLeft -1) / xThreads+1)
	{
		int elementsRemaining = colCount;
		for(int offset = 0; offset+blocksPerGrid_y <= rowCount; offset += blocksPerGrid_y )
		{
			kRowSum <<<dim3(xBlocks,blocksPerGrid_y),dim3(xThreads,1)>>>(dVec,rowCount, colCount, dVec,offset);

			CudaCheckError();
		}

	}
}
template< typename T >
__global__ void FrontFilter(T* vec, bool* front, bool Bool, T val, int vecCount, int pointCount, T* out,int offset)
{
	int vecIndex = blockIdx.x + offset;
	int pointIndex = threadIdx.y + blockIdx.y*blockDim.y;

	if(pointIndex < pointCount )
	{
		if(Bool != front[pointIndex])
		{
			out[vecIndex + pointIndex * vecCount] = val;
		}
		//out[vecIndex + pointIndex * vecCount] = (Bool == front[pointIndex]) ? vec[vecIndex + pointCount*vecCount ] : 0;
	}

}

void PointSelectParetoSerial(bool mode, bool* front, float* scores, int rowCount, int colCount,int _frontCount, int nrem, thrust::host_vector<_teamReward>* toDel)
{

	  struct timeval tv_1;
		struct timeval tv_2;
		struct timezone tz;
		long timer0 = 0;
		gettimeofday(&tv_1, &tz);

	//Sort
	thrust::host_vector<int> hIndex(0);
	//thrust::sequence(hIndex.begin(), hIndex.end());
	thrust::host_vector<float> hScores(0);


	thrust::host_vector<int> hFrontMask(rowCount);
	thrust::device_vector<int> dFrontMask(rowCount);
	int frontCount =0;
	//printf("IDESCAORES: ");
	for(int i=0;i<rowCount;i++)
	{
		if(front[i]== !mode)
		{
			frontCount++;
			hIndex.push_back(i);
			hScores.push_back(scores[i*colCount]);
	//		printf(" %f", scores[i*colCount]);
		//	hFrontMask.push_back(front[i]);
		}

	}
//
//	printf("\nNot Sorted  ");
//		for(int i =0; i <hIndex.size();i++)
//			{
//				printf("  $%d::%f ",hIndex[i], hScores[i]);
//			}
//		printf("\n");
	for(int i= hIndex.size()-1; i > 0 ;i--)
	{
		for(int j=0; j < i;j++)
		{
			if(hScores[j] > hScores[j+1])
			{
				int a = hIndex[j];
				float b = hScores[j];

				hIndex[j] = hIndex[j+1];
				hScores[j] = hScores[j+1] ;

				hIndex[j+1] = a;
				hScores[j+1] = b ;
			}
		}
	}
//printf("\nSoreted  ");
//	for(int i =0; i <hIndex.size();i++)
//		{
//			printf("  $%d::%f ",hIndex[i], hScores[i]);
//		}
//	printf("\n");

//	printf("DELETE");
	for(int i=0; i < nrem;i++)
	{
//		printf("   %d", hIndex[i]);
		toDel->push_back(hIndex[i]);
	}
//	printf("\n");

	//dFrontMask = hFrontMask;
	//int result = thrust::count(dFrontMask.begin(), dFrontMask.end(), 1);

	//printf("FrontCount = %d vs %d",count,result );


//	for(int i=0; i < pointCount;i++)
//	{
//		printf("  %d:%f", i,scores[i]);
//	}
//	printf("\n");




	// Find NLowest
	gettimeofday(&tv_2, &tz);
		timer0 = Diff2(tv_1,tv_2);
//		printf(" Stage Pareto  : %ld\n", timer0 );
}

void PTeamSelectParetoSerial(bool mode, bool* front, float* scores, int rowCount, int colCount,int _frontCount, int nrem, thrust::host_vector<_teamReward>* toDel)
{

	  struct timeval tv_1;
		struct timeval tv_2;
		struct timezone tz;
		long timer0 = 0;
		gettimeofday(&tv_1, &tz);

	//Sort
	thrust::host_vector<int> hIndex(0);
	//thrust::sequence(hIndex.begin(), hIndex.end());
	thrust::host_vector<float> hScores(0);


	thrust::host_vector<int> hFrontMask(rowCount);
	thrust::device_vector<int> dFrontMask(rowCount);
	int frontCount =0;
//	printf("IDESCAORES: ");
	for(int i=0;i<rowCount;i++)
	{
		if(front[i]== !mode)
		{
			frontCount++;
			hIndex.push_back(i);
			hScores.push_back(scores[i*colCount]);
//			printf(" %f", scores[i*colCount]);
		//	hFrontMask.push_back(front[i]);
		}

	}
//
	printf("\nNot Sorted  ");
		for(int i =0; i <hIndex.size();i++)
			{
				printf("  $%d::%f ",hIndex[i], hScores[i]);
			}
		printf("\n");
	for(int i= hIndex.size()-1; i > 0 ;i--)
	{
		for(int j=0; j < i;j++)
		{
			if(hScores[j] > hScores[j+1])
			{
				int a = hIndex[j];
				float b = hScores[j];

				hIndex[j] = hIndex[j+1];
				hScores[j] = hScores[j+1] ;

				hIndex[j+1] = a;
				hScores[j+1] = b ;
			}
		}
	}
printf("\nSoreted  ");
	for(int i =0; i <hIndex.size();i++)
		{
			printf("  $%d::%f ",hIndex[i], hScores[i]);
		}
	printf("\n");

	printf("DELETE");
	for(int i=0; i < nrem;i++)
	{
		printf("   %d", hIndex[i]);
		toDel->push_back(hIndex[i]);
	}
	printf("\n");

	//dFrontMask = hFrontMask;
	//int result = thrust::count(dFrontMask.begin(), dFrontMask.end(), 1);

	//printf("FrontCount = %d vs %d",count,result );


//	for(int i=0; i < pointCount;i++)
//	{
//		printf("  %d:%f", i,scores[i]);
//	}
//	printf("\n");




	// Find NLowest
	gettimeofday(&tv_2, &tz);
		timer0 = Diff2(tv_1,tv_2);
//		printf(" Stage Pareto  : %ld\n", timer0 );
}
__global__ void kElementDivide(int* A, int* B, int vecCount, float* out)
{
	if(threadIdx.x + blockIdx.x*blockDim.x < vecCount)
	out[threadIdx.x + blockIdx.x*blockDim.x] = (A[threadIdx.x +blockIdx.x*blockDim.x]+1) /  (float(B[threadIdx.x + blockIdx.x*blockDim.x])+1);
}

__global__ void kColumnNormalize(int* dData,int* dBaseSum,int rowCount, int colCount, float* dOut)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ int divisor[];

	if(row< rowCount && col < colCount)
	{
	if(threadIdx.y == 0)
	{
		divisor[threadIdx.x] = dBaseSum[col];
	}
	}
	__syncthreads();
	if(row< rowCount && col < colCount)
		{
	dOut[row*colCount+col] = float(dData[row*colCount+col])/divisor[threadIdx.x];
	if(false == isfinite(dOut[row*colCount+col])){ dOut[row*colCount+col] = 0;}
		}

}


__host__ void frontToVector(bool* front,int rowCount,thrust::host_vector<int>* hToDel  )
{
	int count =0;
	for(int i=0;i<rowCount;i++){
		if(!front[i]){
			(*hToDel)[count++]=i;
		}
	}
}

__host__ void calcScores_D(_teamReward* dData, bool* dFront, float* dScores, int rowCount, int colCount ,int frontCount)
{




	int yThreads = 128;
	int yBlocks = (rowCount -1)/ yThreads +1;

	int* dBaseSum;
	int* dASD;


	CudaSafeCall( cudaMalloc( (void**) &dBaseSum,sizeof(int) * colCount * rowCount   ));
	CudaSafeCall(cudaMemcpy (dBaseSum, dData, sizeof(_teamReward) * colCount *rowCount, cudaMemcpyDeviceToDevice));


	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;

	ColumnSum(dBaseSum,rowCount,colCount);

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	int data[rowCount*colCount];
//	CudaSafeCall(cudaMemcpy (data, dData, sizeof(int) * rowCount * colCount, cudaMemcpyDeviceToHost));
//
//	for(int i=0;i < rowCount; i++)
//	{
//		for(int j = 0; j < colCount;j++)
//		{
//			printf(" %d", data[j+i *colCount]);
//		}
//		printf("\n");
//	}
//
//	int distSum[colCount];
//	CudaSafeCall(cudaMemcpy (distSum, dBaseSum, sizeof(int) * colCount, cudaMemcpyDeviceToHost));
//	printf("DataSUM\n");
//	for(int i = 0; i < colCount; i++)
//	{
//		printf(" %d ", distSum[i]);
//	}
//	printf("\n");
//
//	int* dFrontSum;

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int* dWorking;
	CudaSafeCall( cudaMalloc( (void**) &dWorking,sizeof(int) * colCount * rowCount   ));
	CudaSafeCall(cudaMemcpy (dWorking, dData, sizeof(_teamReward) * colCount *rowCount, cudaMemcpyDeviceToDevice));

	int threads = 16;
	kColumnNormalize<<<dim3((colCount-1)/threads+1,(rowCount-1)/threads+1),dim3(threads,threads),threads>>>(dWorking,dBaseSum,rowCount,colCount,dScores);
	CudaCheckError();


	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
	{
	//	printf("adsafghgh \n");
		FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dScores,dFront,0,(float)6.44,colCount, rowCount, dScores, offset);
		CudaCheckError();
	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	float frontSum[colCount*rowCount];
//		CudaSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
//		printf("FORNTSUM\n");
//		for(int i = 0; i < rowCount; i++)
//		{
//			for(int j = 0; j < colCount; j++)
//					{
//			printf(" %0.2f ", frontSum[j + i*colCount ]);
//		}
//		printf("\n");
//		}

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


	RowSum(dScores,rowCount,colCount);
//	for(int _rowsLeft=rowCount; _rowsLeft >1;_rowsLeft = (_rowsLeft -1) / yThreads+1)
//	{
//		printf("rowCount: %d   %d %d\n",rowCount, colCount,yBlocks);
//		int elementsRemaining = colCount;
//		for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
//		{
//			printf("asdasDasd\n");
//			kColumnSum <<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dFrontSum,colCount, rowCount, dFrontSum,offset);
//			CudaCheckMsg("Kernel execution failed");//?? WTF???A
//		}
//
//	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//int frontSum[colCount];
//	CudaSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
//		printf("FORNTSUM\n");
//		for(int j = 0; j < rowCount; j++)
//		{
//			for(int i = 0; i < colCount; i++)
//				printf(" %0.2f ", frontSum[i+j*colCount]);
//			printf("\n");
//		}
//		printf("\n");

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

//	int xThreads = 128;
//	//float* dScore;
//	//CudaSafeCall( cudaMalloc( (void**) &dScore,sizeof(float) *colCount   ));
//	kElementDivide<<<dim3((colCount-1)/xThreads+1,1),dim3(xThreads)>>>(dFrontSum,dBaseSum,colCount,dScores);
//	CudaCheckMsg("Kernel execution failed");

//
//
//	CudaSafeCall(cudaFree( dDistSum));
//	CudaSafeCall(cudaFree( dFrontSum));
//	CudaSafeCall(cudaFree( dScore));
	cudaFree(dWorking);
	cudaFree(dBaseSum);
}


__host__ void calcScores_F(_teamReward* dData, bool* dFront, float* dScores, int rowCount, int colCount ,int frontCount)
{
	int state = STATE_FRONT_TOO_BIG;

	int yThreads = 128;
	int yBlocks = (rowCount -1)/ yThreads +1;

	int* dBaseSum;
	int* dASD;


	CudaSafeCall( cudaMalloc( (void**) &dBaseSum,sizeof(int) * colCount * rowCount   ));
	CudaSafeCall(cudaMemcpy (dBaseSum, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));


	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;
	if(state == STATE_FRONT_TOO_BIG)
	{
	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
		{
			FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dBaseSum,dFront,1,0,colCount, rowCount, dBaseSum, offset);
			CudaCheckError();
		}
	}
	ColumnSum(dBaseSum,rowCount,colCount);



	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	int data[rowCount*colCount];
//	CudaSafeCall(cudaMemcpy (data, dData, sizeof(int) * rowCount * colCount, cudaMemcpyDeviceToHost));
//
//	for(int i=0;i < rowCount; i++)
//	{
//		for(int j = 0; j < colCount;j++)
//		{
//			printf(" %d", data[j+i *colCount]);
//		}
//		printf("\n");
//	}
//
//	int distSum[colCount];
//	CudaSafeCall(cudaMemcpy (distSum, dBaseSum, sizeof(int) * colCount, cudaMemcpyDeviceToHost));
//	printf("DataSUM\n");
//	for(int i = 0; i < colCount; i++)
//	{
//		printf(" %d ", distSum[i]);
//	}
//	printf("\n");
//
//	int* dFrontSum;

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int* dWorking;
	CudaSafeCall( cudaMalloc( (void**) &dWorking,sizeof(int) * colCount * rowCount   ));
	CudaSafeCall(cudaMemcpy (dWorking, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));

	int threads = 16;
	kColumnNormalize<<<dim3((colCount-1)/threads+1,(rowCount-1)/threads+1),dim3(threads,threads),threads>>>(dWorking,dBaseSum,rowCount,colCount,dScores);
	CudaCheckError();
	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
	{
		FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dScores,dFront,STATE_FRONT_TOO_BIG==state,(float)6.44,colCount, rowCount, dScores, offset);
		CudaCheckError();
	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	float frontSum[colCount*rowCount];
//		CudaSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
//		printf("FORNTSUM\n");
//		for(int i = 0; i < rowCount; i++)
//		{
//			for(int j = 0; j < colCount; j++)
//					{
//			printf(" %0.2f ", frontSum[j + i*colCount ]);
//		}
//		printf("\n");
//		}

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	RowSum(dScores,rowCount,colCount);

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//int frontSum[colCount];
//	CudaSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
//		printf("FORNTSUM\n");
//		for(int j = 0; j < rowCount; j++)
//		{
//			for(int i = 0; i < colCount; i++)
//				printf(" %0.2f ", frontSum[i+j*colCount]);
//			printf("\n");
//		}
//		printf("\n");

	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	cudaFree(dWorking);
	cudaFree(dBaseSum);
}


__host__ int findFront( _teamReward* dData,int rowCount, int colCount, bool* dFront , int gapSize)
{


	bool* dTmpDistDom;
	bool* dTmpDistEqual;
	CudaSafeCall( cudaMalloc( (void**) &dTmpDistDom,sizeof(bool) * rowCount *rowCount *3));
	CudaSafeCall( cudaMalloc( (void**) &dTmpDistEqual,sizeof(bool) *rowCount * rowCount *3 ));


	int* dTMP;
	CudaSafeCall( cudaMalloc( (void**) &dTMP,sizeof(int) *rowCount * rowCount));
	CudaCheckError();
	int hTMP[rowCount*rowCount];
	CudaSafeCall(cudaMemcpy (hTMP, dTMP,sizeof(int) * rowCount*rowCount , cudaMemcpyDeviceToHost));
//	_teamReward tmp[rowCount*colCount];
//	CudaSafeCall(cudaMemcpy (tmp, dData,sizeof(_teamReward) * rowCount*colCount , cudaMemcpyDeviceToHost));
	CudaCheckError();
//	printf(" %d %p  %d  %d  %p  %p   %p\n",rowCount,dData,colCount,rowCount,dTmpDistDom,dTmpDistEqual,dTMP);

	isDominated<<<dim3(rowCount,rowCount),dim3(128)>>>(dData,colCount,rowCount,dTmpDistDom,dTmpDistEqual, dTMP);
	CudaCheckError();
	//	CudaSafeCall(cudaMemcpy (hTMP, dTMP,sizeof(int) * rowCount*rowCount , cudaMemcpyDeviceToHost));
//
//	printf("##$$##$$##$$\n");
//	for(int i=0;i < rowCount;i++)
//	{
//		for(int j=0;j < rowCount;j++)
//			{
//				printf(" %d", hTMP[i*rowCount+j]);
//			}
//		printf("\n");
//	}
//

	FindParetoFront(dTmpDistDom,dFront,rowCount);
	bool front[rowCount];
	CudaSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

	int frontCount =0;
	for(int i=0; i< rowCount;i++)
		{
			if(front[i])
			{
				frontCount++;
			}
		}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
	printf("selPoints F: ");
	for(int i=0; i< rowCount;i++)
	{
		if(front[i])
		{
			printf(" %d", i);
		//	frontCount++;
		}
	}
	printf("\nselPoints D: ");
	for(int i=0; i< rowCount;i++)
	{
		if(!front[i])
			printf(" %d", i);
	}
	printf("\n");
//	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	 cudaFree( dTmpDistDom);
	 cudaFree( dTmpDistEqual);

	 cudaFree(dTMP);


		return frontCount;
}
__host__ void SelectTeamsGPU(int rowCount, int colCount, _teamReward* data, int gapSize, thrust::host_vector<_teamReward>* hToDel )
{
	printf("SELECT TEAMS\n");
	_teamReward* dData;
	CudaSafeCall( cudaMalloc( (void**) &dData,sizeof(_teamReward) * rowCount*colCount     ));
	CudaSafeCall(cudaMemcpy (dData,data, sizeof(_teamReward) * rowCount*colCount   , cudaMemcpyHostToDevice));
	CudaCheckError();

	bool* dFront;
	CudaSafeCall( cudaMalloc( (void**) &dFront,sizeof(bool) * rowCount ));

	int frontCount = findFront(dData,rowCount,colCount,dFront, gapSize);

	printf(" Keep = Rowcount:%d - gapSize:%d", rowCount, gapSize);
	int keep = rowCount - gapSize;
	if(frontCount == keep)
		{
			printf(" F == SAME\n");

			bool front[rowCount];
			CudaSafeCall(cudaMemcpy (front,dFront, sizeof(bool) * rowCount   , cudaMemcpyDeviceToHost));
			hToDel->clear();
			for(int i=0;i<rowCount;i++)
			{
				if(!front[i])
				{
					hToDel->push_back(i);
				}
			}
		//	frontToVector(front,rowSize,hToDel);
		}
		else{

			float* dScores;
			CudaSafeCall( cudaMalloc( (void**) &dScores,sizeof(float) * rowCount*colCount ));

			if (frontCount < keep)
			{
				printf(" F == TOO SMALL\n");
				calcScores_D(dData,dFront, dScores,rowCount,colCount,frontCount);

				bool front[rowCount];
				CudaSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

				float scores[rowCount*colCount];
				CudaSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
				hToDel->clear();
				PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize,hToDel);
			}else{
				printf(" F == TOO BIG\n");
				calcScores_F(dData,dFront, dScores,rowCount,colCount,frontCount);

				bool front[rowCount];
				CudaSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

				float scores[rowCount*colCount];
				CudaSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
				hToDel->clear();
				for(int i=0;i < rowCount;i++)
				{
					if(!front[i]){
						hToDel->push_back(i);
					}
				}
printf("###### , V:%d   = gap:%d - ( rowCount:%d - frontFoucnt:%d)\n",gapSize-(rowCount-frontCount),gapSize,rowCount,frontCount);
				PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize-(rowCount-frontCount),hToDel);


				//PselectParetoSerial(frontCount < keep, front,scores, rowCount, 0, gapSize,hToDel);
			}
			cudaFree(dScores);
		}
		cudaFree(dData);
		cudaFree(dFront);
		printf("TEAM SELECT END\n");
}

__host__ void SelectPointsGPU(int rowCount, int colCount, _teamReward* data, int gapSize, thrust::host_vector<_teamReward>* hToDel )
{

	struct timeval tv_start;
	struct timeval tv_end;
	struct timezone tz;
	long timerDist =0;

	int vecCount = colCount*colCount;
	_teamReward* dDist;

	//for(int i=0;i<20; i++) { data[i*rowCount] = i%10;}
//	printf(" BALLLLS\n");
//	for(int i =0; i < colCount;i++)
//	{
//		for(int j=0;j < rowCount;j++)
//		{
//			printf("%d,", data[i*rowCount+j]);
//		}
//	printf("\n");
//	}
//
//	printf("%d = %d * %d * %d\n" ,vecCount*rowCount , colCount,colCount,rowCount  );

	//pivot
	_teamReward pivotData[rowCount*colCount];
	for(int i=0; i<rowCount ;i++)
	{
	  for(int j=0; j < colCount;j++)
	  {
	     pivotData[j*rowCount+i] =  data[i*colCount+j];
	  }
	}


	CudaSafeCall( cudaMalloc( (void**) &dDist,sizeof(_teamReward) * vecCount*rowCount ));
//	printf("@Point FindFront Row:%d Col:%d \n", rowCount,vecCount);
	bool* dFront;
	CudaSafeCall( cudaMalloc( (void**) &dFront,sizeof(bool) * rowCount ));

	_teamReward* dData;
	CudaSafeCall( cudaMalloc( (void**) &dData,sizeof(_teamReward) * rowCount*colCount     ));
	CudaSafeCall(cudaMemcpy (dData,pivotData, sizeof(_teamReward) * rowCount*colCount   , cudaMemcpyHostToDevice));

	 gettimeofday(&tv_start, &tz);
	 GetDist(dData,dDist,colCount,rowCount);
	 gettimeofday(&tv_end, &tz);
	 timerDist = Diff2(tv_start,tv_end);



 //   printf("PSTAT %ld\n",timerDist);
	//GetDistTex(data,dDist,rowCount,colCount);
//	kCalcDist<<<dim3(rowCount,1),dim3(512)>>>(dData, dDist, colCount, rowCount);
	CudaCheckError();

//	//	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	_teamReward* dist_out = (_teamReward*) malloc(sizeof(_teamReward) * vecCount*rowCount);
//	CudaSafeCall(cudaMemcpy (dist_out,dDist, sizeof(_teamReward) *  vecCount*rowCount/*teamCount*teamCount*pointCount*/ , cudaMemcpyDeviceToHost));
//	printf("selPoints distinctions\n");
//	for(int i = 0; i < rowCount;i++)
//	{
//		for(int j = 0; j < vecCount;j++)
//		{
//			printf("%d", dist_out[j+ i*vecCount]);
//			if(j%colCount == colCount-1){printf(" ");}
//		}
//		printf("\n");
//	}
//	printf("\n");
//
//	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


//		printf("Point FindFront Row:%d Col:%d \n", rowCount,vecCount);
	int frontCount = findFront(dDist,rowCount,vecCount,dFront, gapSize);
	int keep = rowCount - gapSize;
//	printf("GAPSTATS RowCount:%d GapSize:%d  Keep:%d   FrontCount:%d\n", rowCount,gapSize, keep , frontCount);

//	printf("T:%d G:%d K:%d F:%d\n", rowCount,gapSize,keep,frontCount);
	if(frontCount == keep)
	{
//		printf(" F == SAME\n");

		bool front[rowCount];
		CudaSafeCall(cudaMemcpy (front,dFront, sizeof(bool) * rowCount   , cudaMemcpyDeviceToHost));
		hToDel->clear();
		for(int i=0;i<rowCount;i++)
		{
			if(!front[i])
			{
				hToDel->push_back(i);
			}
		}
	//	frontToVector(front,rowSize,hToDel);
	}
	else{

		float* dScores;
		CudaSafeCall( cudaMalloc( (void**) &dScores,sizeof(float) * rowCount*colCount ));

		if (frontCount < keep)
		{
//			printf(" F == TOO SMALL\n");
			calcScores_D(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
			CudaSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

			float scores[rowCount*colCount];
			CudaSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
			hToDel->clear();
			PointSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize,hToDel);
		}else{
//			printf(" F == TOO BIG\n");
			calcScores_F(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
			CudaSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

			float scores[rowCount*colCount];
			CudaSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
			hToDel->clear();
			for(int i=0;i < rowCount;i++)
			{
				if(!front[i]){
					hToDel->push_back(i);
				}
			}
//			printf("HTP SIZE: %d\n", hToDel->size());
			PointSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize-(rowCount-frontCount),hToDel);


			//PselectParetoSerial(frontCount < keep, front,scores, rowCount, 0, gapSize,hToDel);
		}
		cudaFree(dScores);
	}
	cudaFree(dData);
	cudaFree(dFront);
	cudaFree(dDist);

//	printf(" Leave: PSel\n");
}






