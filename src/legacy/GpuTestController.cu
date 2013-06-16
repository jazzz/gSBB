#define _DEVICE_VARS_TEST_
#include "GpuTestController.cuh"



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


#define MAXTHREADS 256


__global__ void kLearnerTest(int val,
                      int learnOffset,
                      int pointOffset,
                      _learner* dLearnerMatrix,
                      _learnerBid* dLearnerBidMatrix,
                      _point* dPointMatrix,
                      int learnerCount,
                      int pointCount)
{
    int pointId = threadIdx.x + blockIdx.x * blockDim.x + pointOffset; // + (blockIdx.x*gridDim.x);
    int learnerId = threadIdx.y + blockIdx.y* blockDim.y + learnOffset;// + blockIdx.y * blockDim.y;
    if (learnerId < TOTAL_LEARNERS && pointId < TOTAL_POINTS)
    {
        int id = threadIdx.x * blockDim.y + threadIdx.y;
        //_learner *shared_learner = &dLearnerMatrix[(threadIdx.y + blockIdx.y*blockDim.y) * LEARNER_LENGTH];
        //_point *feature = &dPointMatrix[(threadIdx.x + blockIdx.x *blockDim.x)*NUM_FEATURES];

        _learner *shared_learner = &dLearnerMatrix[learnerId*LEARNER_LENGTH];
        _point *feature = &dPointMatrix[pointId*NUM_FEATURES];

        //_learner *shared_learner = &dLearnerMatrix[];
        //_point *feature = &dPointMatrix[0];

        __shared__ _learnerBid registers[MAXTHREADS][8];

        registers[id][0] =0;
        registers[id][1] =0;
        registers[id][2] =0;
        registers[id][3] =0;
        registers[id][4] =0;
        registers[id][5] =0;
        registers[id][6] =0;
        registers[id][7] =0;

        //short progsize = shared_learner[0];
        for (int i=1;i<=shared_learner[0];i++)
        {
            _learnerBid* dst = &registers[id][((shared_learner[i] & DST_MASK) >> DST_SHIFT)];

            _learnerBid srcVal;

            if (1 == ((shared_learner[i] & MODE_MASK) >> MODE_SHIFT ) %2) {

               // srcVal =  dPointMatrix[threadIdx.x* NUM_FEATURES + ((shared_learner[i] & SRC_MASK) >> SRC_SHIFT) % NUM_FEATURES ];
                srcVal =  feature[((shared_learner[i] & SRC_MASK) >> SRC_SHIFT) % NUM_FEATURES ];
            }else{
                srcVal =     registers[id][(((shared_learner[i] & SRC_MASK) >> SRC_SHIFT) % REG_COUNT)];
            }
            switch ( ((shared_learner[i] & OP_MASK) >> OP_SHIFT) % OP_CODE_COUNT){
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

        dLearnerBidMatrix[ threadIdx.y * blockDim.x +  blockIdx.x*blockDim.x + threadIdx.x] =  1 / (1+exp(-registers[id][0]));
         // dLearnerBidMatrix[ threadIdx.y * blockDim.x +  blockIdx.x*blockDim.x + threadIdx.x] = val+1;
        }

}

__global__ void kEval2(int val,
                      int learnOffset,
                      int pointOffset,
                      _learner* dLearnerMatrix,
                      _learnerBid* dLearnerBidMatrix,
                      _point* dPointMatrix,
                      int learnerCount,
                      int pointCount)
{
    int pointId = threadIdx.x + blockIdx.x * blockDim.x + pointOffset ;
    int learnerId = threadIdx.y + blockIdx.y* blockDim.y + learnOffset ;
   // int learnerId = threadIdx.y  + blockIdx.y * blockDim.y ;
    if (learnerId < TOTAL_LEARNERS && pointId < TOTAL_POINTS)
    {
      //   dLearnerBidMatrix[ learnerId*TOTAL_POINTS + pointId] = dPointMatrix[pointId * NUM_FEATURES + learnerId] ; // + min(LEARNER_LENGTH-1, pointId)];// blockIdx.x * gridDim.y + blockIdx.y + 1;

      //  dLearnerBidMatrix[ learnerId*TOTAL_POINTS + pointId] = 1 ;//learnOffset;//blockIdx.x * gridDim.y + blockIdx.y + 1;
        dLearnerBidMatrix[ threadIdx.y * blockDim.x +  threadIdx.x] = threadIdx.x;//blockIdx.x * gridDim.y + blockIdx.y + 1;

    }
}

__host__
void TestLearners(	int learnerCount,
					int learnerLength,
					_learner* hLearnerMatrix,
					_learnerBid* hLearnerBidMatrix,
					int trainingSetSize,
					_point* hTrainingSet ,
					int numFeatures
				 )
{

	int bytesize_singlePoint = sizeof(_point) * numFeatures;
	int bytesize_learner = sizeof(_learner) * learnerLength;
    //////////////////////////
	// Memory Allocations
	//////////////////////////
	int streamCount = 2;


	_point* dTrainingSet;
	_learner* dLearnerMatrix;
	_learnerBid* dBidMatrix[streamCount];

	 int learnersPerChunk =1 ;
	 int pointsPerChunk = 512;


	 cutilSafeCall( cudaMalloc( (void**) &dTrainingSet, bytesize_singlePoint * trainingSetSize ));
	 cutilSafeCall( cudaMalloc( (void**) &dLearnerMatrix, bytesize_learner * learnerCount ));

	 cudaStream_t* stream = new cudaStream_t[streamCount ];
	 	for(int i=0; i < streamCount; i++)
	 	{
	 		cutilSafeCall( cudaMalloc( (void**) &dBidMatrix[i], sizeof(_learnerBid) * learnersPerChunk* pointsPerChunk ));
	 		cudaStreamCreate(&stream[i]);
	 	}

	// cutilSafeCall( cudaMalloc( (void**) &dBidMatrix[1], sizeof(_learnerBid) * learnersPerChunk* pointsPerChunk ));

	 cudaMemset(dLearnerMatrix,0,bytesize_learner * learnerCount );
	 cudaMemset(dBidMatrix[0],0,sizeof(_learnerBid) * learnersPerChunk * pointsPerChunk  );

    cutilSafeCall( cudaMemcpyToSymbol( TOTAL_LEARNERS, &learnerCount,sizeof(short)));
    cutilSafeCall( cudaMemcpyToSymbol( TOTAL_POINTS, &trainingSetSize,sizeof(short)));

    cutilSafeCall( cudaMemcpyToSymbol( NUM_FEATURES, &numFeatures,sizeof(short)));
    cutilSafeCall( cudaMemcpyToSymbol( LEARNER_LENGTH, &learnerLength,sizeof(short)));


     int learnerChunkCount = (learnerCount-1)/ learnersPerChunk+1;
     int pointChunkCount = (trainingSetSize-1) / pointsPerChunk+1;

    int pointsPerBlock = 256;
     int learnersPerBlock = min(MAXTHREADS/pointsPerBlock , learnerCount);
     int threadsPerBlock_x = pointsPerBlock ;
     int threadsPerBlock_y = learnersPerBlock;



     int blocksPerGrid_x = ((pointsPerChunk-1)/pointsPerBlock+1);
     int blocksPerGrid_y = ((learnersPerChunk-1)/learnersPerBlock+1);

     dim3 GRID (blocksPerGrid_x,blocksPerGrid_y);
     dim3 BLOCK (threadsPerBlock_x,threadsPerBlock_y);



  //   printf(" ###  lpC:%d   lCC:%d     ppC:%d     pCC:%d  PS:%d   BPGx%d   BPGy:%d\n" , learnersPerChunk , learnerChunkCount, pointsPerChunk, pointChunkCount , trainingSetSize, blocksPerGrid_x ,blocksPerGrid_y );

    cutilSafeCall(cudaMemcpy (dLearnerMatrix, hLearnerMatrix, learnerCount* bytesize_learner, cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy (dTrainingSet, hTrainingSet, trainingSetSize* bytesize_singlePoint, cudaMemcpyHostToDevice));
//

    int streamA = 0;
    int streamB = 1;

    for(int chunkId_y=0; chunkId_y< learnerChunkCount;chunkId_y ++)
    {

     	 for(int chunkId_x=0; chunkId_x< pointChunkCount;chunkId_x +=streamCount)
          {
     		 //kLearnerTest<<<GRID,BLOCK,0,stream[streamId]>>>(chunkId_x,chunkId_y        ,chunkId_x*pointsPerChunk,dLearnerMatrix,dBidMatrix[0], dTrainingSet, 1, trainingSetSize);
     		 for(int streamId=0; streamId < streamCount; streamId++)
     		 {
     			 if(chunkId_x+streamId < pointChunkCount){
     				 kLearnerTest<<<GRID,BLOCK,0,stream[streamId]>>>(chunkId_x+streamId,chunkId_y        ,(chunkId_x+streamId)*pointsPerChunk,dLearnerMatrix,dBidMatrix[streamId], dTrainingSet, 1, trainingSetSize);
     			 }
     		 }

     		for(int streamId=0; streamId < streamCount; streamId++)
     		     		 {
     		     			 if(chunkId_x+streamId < pointChunkCount){
     		     				cutilSafeCall(cudaMemcpyAsync (hLearnerBidMatrix + chunkId_y*trainingSetSize + (chunkId_x+streamId)* pointsPerChunk , dBidMatrix[streamId], pointsPerChunk * sizeof(_learnerBid), cudaMemcpyDeviceToHost,stream[streamId]));
     		     			 }
     		     		 }

          }
     	// int offset = chunkId_y * learnersPerChunk * trainingSetSize;
  		//cutilSafeCall(cudaMemcpyAsync (hLearnerBidMatrix + chunkId_y*trainingSetSize , dBidMatrix[0], trainingSetSize * sizeof(_learnerBid), cudaMemcpyDeviceToHost,stream[streamB]));


      }

	cudaDeviceSynchronize();

    cutilSafeCall( cudaFree( dLearnerMatrix));
    cutilSafeCall( cudaFree( dBidMatrix[0]));
    //cutilSafeCall( cudaFree( dBidMatrix[1]));
    cutilSafeCall( cudaFree( dTrainingSet));

    for(int i=0; i < streamCount; i++)
    	{
    		cudaStreamDestroy(stream[i]);
    	}

}
