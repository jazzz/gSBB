#include "CudaControllerFunc.cuh"
#include "CudaControllerVars.cuh"

#define _DEVICE_VARS_

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
    if (learnerId < TOTAL_LEARNERS && pointId < TOTAL_POINTS)
    {
		int id = threadIdx.x*blockDim.y + threadIdx.y;
		_learner *learner = &dLearnerMatrix[(threadIdx.y + blockIdx.y*blockDim.y) * LEARNER_LENGTH];
        _point *feature = &dPointMatrix[(threadIdx.x+blockDim.x*blockIdx.x)/PIVOT_STRIPE_SIZE * PIVOT_STRIPE_SIZE* NUM_FEATURES + ((threadIdx.x+blockDim.x*blockIdx.x)% PIVOT_STRIPE_SIZE) ];
      //_learner *shared_learner = &dLearnerMatrix[];
        //_point *feature = &dPointMatrix[0];

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


__host__
void EvaluateLearners(_learner* hLearnerMatrix, _learnerBid* dLearnerBidMatrix,_point* dPointMatrix, int learnerCount, int pointCount)
{


	int xThreads = 16;
	int yThreads = 16;

	int xBlocks = 16;
	int yBlocks = 16;


	_learner* dLearnerMatrix;
    cutilSafeCall( cudaMalloc( (void**) &dLearnerMatrix, bytesize_learner * learnerCount   ));
	cutilSafeCall(cudaMemcpy (dLearnerMatrix, dLearnerMatrix, bytesize_learner * learnerCount  , cudaMemcpyHostToDevice));

	kLearnerEval<<<dim3(xBlocks,yBlocks),dim3(xThreads,yThreads)>>>(0,0,dLearnerMatrix,dLearnerBidMatrix,dPointMatrix,learnerCount, pointCount);



	cudaFree(dLearnerMatrix);
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





