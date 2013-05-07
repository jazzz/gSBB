#define _DEVICE_VARS_

#define DEBUGMODE
#include "GpuController_Selection.cuh"

  #include <thrust/logical.h>
  #include <thrust/functional.h>

long Diff2(timeval tv_start, timeval tv_end){
   return 1000000*(tv_end.tv_sec - tv_start.tv_sec) + tv_end.tv_usec - tv_start.tv_usec;
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
__global__ void kCalcDistSoMuchBetterBro(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
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


__host__ void GetDist(_teamReward* rewards, _teamReward* dist_out, int teamCount, int pointCount)
{
	int threads = 8;
	int xThreads = threads;
	int yThreads = threads;

	int xBlocks = (teamCount-1) / xThreads +1;
	int yBlocks = (teamCount-1) / yThreads +1;
	int zBlocks = pointCount;


	cudaMemset(dist_out,0,sizeof(_teamReward) * teamCount*teamCount*pointCount);

	if(xThreads*yThreads > 1024){ fprintf(stderr,"Error: Too many threads used in GetDist");}
	kCalcDistSoMuchBetterBro<<<dim3(xBlocks,yBlocks,zBlocks),dim3(xThreads,yThreads),xThreads+yThreads>>>(rewards, dist_out, teamCount, pointCount);
	cutilCheckMsg("Kernel execution failed");//?? WTF???A



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


	cutilCheckMsg("Pre Kernel execution failed");//?? WTF???A

//	bool* A = new bool[rowCount*colCount];
//	cutilSafeCall(cudaMemcpy (A, dom, sizeof(bool) * colCount * rowCount, cudaMemcpyDeviceToHost));
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
	cutilSafeCall( cudaMalloc( (void**) &dStaging,sizeof(bool) * rowCount * xBlocks   ));
	int* tmp;
		cutilSafeCall( cudaMalloc( (void**) &tmp,sizeof(int) * rowCount * rowCount   ));
	//	cutilSafeCall(cudaMemcpy (dWorking, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));

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
			cutilCheckMsg("Kernel execution failed");//?? WTF???A


			rowOffset += blocksPerGrid_y;
		}
		colsLeft = xBlocks;
	}
		int t[rowCount*rowCount];
//		cutilSafeCall(cudaMemcpy (t, tmp, sizeof(int) * rowCount *rowCount, cudaMemcpyDeviceToHost));
//		cutilSafeCall(cudaMemcpy (A, dStaging, sizeof(bool) * xBlocks *rowCount, cudaMemcpyDeviceToHost));
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
//	cutilSafeCall(cudaMemcpy (front, dStaging, sizeof(bool) *rowCount, cudaMemcpyDeviceToDevice));

//	cutilSafeCall(cudaMemcpy (A, front, sizeof(bool) * rowCount, cudaMemcpyDeviceToHost));
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
	printf("ColSum - R:%d C:%d ", rowCount, colCount);
	int yThreads = 128;
	int yBlocks = (rowCount -1)/ yThreads +1;
	cutilCheckMsg("Pre Kernel execution failed");//?? WTF???A
	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;
	for(int _rowsLeft=rowCount; _rowsLeft >1;_rowsLeft = (_rowsLeft -1) / yThreads+1)
	{
		printf("rowCount: %d   %d %d\n",rowCount, colCount,yBlocks);
		int elementsRemaining = colCount;
		for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
		{
			printf("asdasDasd %d\n", offset);
			kColumnSum <<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(vec,colCount, rowCount, vec,offset);
			cutilCheckMsg("Kernel execution failed");//?? WTF???A
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
	printf("RowSum - R:%d C:%d ", rowCount, colCount);
	int xThreads = 128;
	int xBlocks =  (colCount -1)/ xThreads +1;
	int blocksPerGrid_y = (rowCount > 40) ? 40 : rowCount;
	for(int _rowsLeft=colCount; _rowsLeft >1;_rowsLeft = (_rowsLeft -1) / xThreads+1)
	{
		printf(" XB:%d XT:%d\n",xBlocks, xThreads);
		int elementsRemaining = colCount;
		for(int offset = 0; offset+blocksPerGrid_y <= rowCount; offset += blocksPerGrid_y )
		{
			printf("asdasDasd\n");
			kRowSum <<<dim3(xBlocks,blocksPerGrid_y),dim3(xThreads,1)>>>(dVec,rowCount, colCount, dVec,offset);

			cutilCheckMsg("Kernel execution failed");//?? WTF???A
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



void PTeamSelectParetoSerial(bool mode, bool* front, float* scores, int rowCount, int colCount,int _frontCount, int nrem, thrust::host_vector<int>* toDel)
{

	  struct timeval tv_1;
		struct timeval tv_2;
		struct timezone tz;
		long timer0 = 0;
		gettimeofday(&tv_1, &tz);
	//printf("SIZE TODEL: %d\n", toDel->size());
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
	//printf("\nNot Sorted  ");
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

	for(int i=0; i < nrem;i++)
	{
		toDel->push_back(hIndex[i]);
	}

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
		printf(" Stage Pareto  : %ld\n", timer0 );
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


	cutilSafeCall( cudaMalloc( (void**) &dBaseSum,sizeof(int) * colCount * rowCount   ));
	cutilSafeCall(cudaMemcpy (dBaseSum, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));


	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;

	ColumnSum(dBaseSum,rowCount,colCount);

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	int data[rowCount*colCount];
//	cutilSafeCall(cudaMemcpy (data, dData, sizeof(int) * rowCount * colCount, cudaMemcpyDeviceToHost));
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
//	cutilSafeCall(cudaMemcpy (distSum, dBaseSum, sizeof(int) * colCount, cudaMemcpyDeviceToHost));
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
	cutilSafeCall( cudaMalloc( (void**) &dWorking,sizeof(int) * colCount * rowCount   ));
	cutilSafeCall(cudaMemcpy (dWorking, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));

	int threads = 16;
	kColumnNormalize<<<dim3((colCount-1)/threads+1,(rowCount-1)/threads+1),dim3(threads,threads),threads>>>(dWorking,dBaseSum,rowCount,colCount,dScores);



	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
	{
	//	printf("adsafghgh \n");
		FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dScores,dFront,0,(float)6.44,colCount, rowCount, dScores, offset);
		cutilCheckMsg("Kernel execution failed");
	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	float frontSum[colCount*rowCount];
//		cutilSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
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
//			cutilCheckMsg("Kernel execution failed");//?? WTF???A
//		}
//
//	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//int frontSum[colCount];
//	cutilSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
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
//	//cutilSafeCall( cudaMalloc( (void**) &dScore,sizeof(float) *colCount   ));
//	kElementDivide<<<dim3((colCount-1)/xThreads+1,1),dim3(xThreads)>>>(dFrontSum,dBaseSum,colCount,dScores);
//	cutilCheckMsg("Kernel execution failed");

//
//
//	cutilSafeCall(cudaFree( dDistSum));
//	cutilSafeCall(cudaFree( dFrontSum));
//	cutilSafeCall(cudaFree( dScore));
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


	cutilSafeCall( cudaMalloc( (void**) &dBaseSum,sizeof(int) * colCount * rowCount   ));
	cutilSafeCall(cudaMemcpy (dBaseSum, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));


	int blocksPerGrid_x = (colCount > 40) ? 40 : colCount;
	if(state == STATE_FRONT_TOO_BIG)
	{
	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
		{
			FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dBaseSum,dFront,1,0,colCount, rowCount, dBaseSum, offset);
			cutilCheckMsg("Kernel execution failed");
		}
	}
	ColumnSum(dBaseSum,rowCount,colCount);



	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	int data[rowCount*colCount];
//	cutilSafeCall(cudaMemcpy (data, dData, sizeof(int) * rowCount * colCount, cudaMemcpyDeviceToHost));
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
//	cutilSafeCall(cudaMemcpy (distSum, dBaseSum, sizeof(int) * colCount, cudaMemcpyDeviceToHost));
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
	cutilSafeCall( cudaMalloc( (void**) &dWorking,sizeof(int) * colCount * rowCount   ));
	cutilSafeCall(cudaMemcpy (dWorking, dData, sizeof(int) * colCount *rowCount, cudaMemcpyDeviceToDevice));

	int threads = 16;
	kColumnNormalize<<<dim3((colCount-1)/threads+1,(rowCount-1)/threads+1),dim3(threads,threads),threads>>>(dWorking,dBaseSum,rowCount,colCount,dScores);

	for(int offset = 0; offset+blocksPerGrid_x <= colCount; offset += blocksPerGrid_x )
	{
		FrontFilter<<<dim3(blocksPerGrid_x,yBlocks),dim3(1,yThreads)>>>(dScores,dFront,STATE_FRONT_TOO_BIG==state,(float)6.44,colCount, rowCount, dScores, offset);
		cutilCheckMsg("Kernel execution failed");
	}

	// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	float frontSum[colCount*rowCount];
//		cutilSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
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
//	cutilSafeCall(cudaMemcpy (frontSum, dScores, sizeof(float) * colCount * rowCount, cudaMemcpyDeviceToHost));
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
	cutilSafeCall( cudaMalloc( (void**) &dTmpDistDom,sizeof(bool) * rowCount *rowCount ));
	cutilSafeCall( cudaMalloc( (void**) &dTmpDistEqual,sizeof(bool) *rowCount * rowCount  ));


	int* dTMP;
	cutilSafeCall( cudaMalloc( (void**) &dTMP,sizeof(int) *rowCount * rowCount));


	_teamReward hData[rowCount*colCount];
	cutilSafeCall(cudaMemcpy (hData, dData,sizeof(_teamReward) * rowCount*colCount , cudaMemcpyDeviceToHost));
//	printf(">>>>DATA===\n");
//	for(int i=0;i < rowCount;i++)
//	{
//		for(int j=0;j < colCount;j++)
//			{
//				printf(" %d", hData[i*colCount+j]);
//			}
//		printf("\n");
//	}


//	int* hTMP = (int*)malloc(sizeof(int) * rowCount*rowCount);
//	for(int i=0; i < rowCount*rowCount; i++) { hTMP[i] =8;}
//	cutilSafeCall(cudaMemcpy (dTMP, hTMP,sizeof(int) * rowCount*rowCount , cudaMemcpyHostToDevice));


//	printf(" %d \n", rowCount);
	isDominated<<<dim3(rowCount,rowCount),dim3(128)>>>(dData,colCount,rowCount,dTmpDistDom,dTmpDistEqual, dTMP);
//	cutilSafeCall(cudaMemcpy (hTMP, dTMP,sizeof(int) * rowCount*rowCount , cudaMemcpyDeviceToHost));
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
	cutilSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

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
	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	 cudaFree( dTmpDistDom);
	 cudaFree( dTmpDistEqual);

	 cudaFree(dTMP);


		return frontCount;
}
__host__ void SelectTeamsGPU(int rowCount, int colCount, _teamReward* data, int gapSize, thrust::host_vector<int>* hToDel )
{

	printf(" Start: TSel\n");




	bool* dFront;
	cutilSafeCall( cudaMalloc( (void**) &dFront,sizeof(bool) * rowCount ));

	_teamReward* dData;
	cutilSafeCall( cudaMalloc( (void**) &dData,sizeof(_teamReward) * rowCount*colCount     ));
	cutilSafeCall(cudaMemcpy (dData,data, sizeof(_teamReward) * rowCount*colCount   , cudaMemcpyHostToDevice));


	bool front[rowCount];
	int frontCount = findFront(dData,rowCount,colCount,dFront, gapSize);
	cutilSafeCall(cudaMemcpy (front,dFront, sizeof(bool) * rowCount   , cudaMemcpyDeviceToHost));

	printf(" TF: ");
	for(int i=0;i<rowCount;i++)
		{
			if(front[i])
			{
				printf("    %d",i);
			}
		}
	printf("\n TD: ");
	for(int i=0;i<rowCount;i++)
		{
			if(!front[i])
			{
				printf("    %d",i);
			}
		}
	printf("\n\n");
	int keep = rowCount - gapSize;
	//printf("T:%d G:%d K:%d F:%d\n", rowCount,gapSize,keep,frontCount);
	if(frontCount == keep)
	{
		printf(" F == SAME\n");

		bool front[rowCount];
		cutilSafeCall(cudaMemcpy (front,dFront, sizeof(bool) * rowCount   , cudaMemcpyDeviceToHost));
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
		cutilSafeCall( cudaMalloc( (void**) &dScores,sizeof(float) * rowCount*colCount ));

		if (frontCount < keep)
		{
			printf(" F == TOO SMALL\n");
			calcScores_D(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
			cutilSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

			float scores[rowCount*colCount];
			cutilSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
			hToDel->clear();
			PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize,hToDel);
		}else{
			printf(" F == TOO BIG\n");
			calcScores_F(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
					cutilSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

					float scores[rowCount*colCount];
					cutilSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));

					hToDel->clear();
					for(int i=0;i < rowCount;i++)
															{
																if(!front[i]){
																	hToDel->push_back(i);
																}
															}
					PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0,gapSize - (rowCount - frontCount),hToDel);



				//PselectParetoSerial(frontCount < keep, front,scores, rowCount, 0, gapSize,hToDel);
		}
		cudaFree(dScores);
	}
	cudaFree(dData);
	cudaFree(dFront);


}

__host__ void SelectPointsGPU(int rowCount, int colCount, _teamReward* data, int gapSize, thrust::host_vector<_teamReward>* hToDel )
{
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

	cutilSafeCall( cudaMalloc( (void**) &dDist,sizeof(_teamReward) * vecCount*rowCount ));

	bool* dFront;
	cutilSafeCall( cudaMalloc( (void**) &dFront,sizeof(bool) * rowCount ));

	_teamReward* dData;
	cutilSafeCall( cudaMalloc( (void**) &dData,sizeof(_teamReward) * rowCount*colCount     ));
	cutilSafeCall(cudaMemcpy (dData,data, sizeof(_teamReward) * rowCount*colCount   , cudaMemcpyHostToDevice));

	GetDist(dData,dDist,colCount,rowCount);
//	kCalcDist<<<dim3(rowCount,1),dim3(512)>>>(dData, dDist, colCount, rowCount);
//	cutilCheckMsg("Kernel execution failed");

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	_teamReward* dist_out = (_teamReward*) malloc(sizeof(_teamReward) * vecCount*rowCount);
//	cutilSafeCall(cudaMemcpy (dist_out,dDist, sizeof(_teamReward) *  vecCount*rowCount/*teamCount*teamCount*pointCount*/ , cudaMemcpyDeviceToHost));
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

//	// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


	int frontCount = findFront(dDist,rowCount,vecCount,dFront, gapSize);

	int keep = rowCount - gapSize;
	printf("T:%d G:%d K:%d F:%d\n", rowCount,gapSize,keep,frontCount);
	if(frontCount == keep)
	{
		printf(" F == SAME\n");

		bool front[rowCount];
		cutilSafeCall(cudaMemcpy (front,dFront, sizeof(bool) * rowCount   , cudaMemcpyDeviceToHost));
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
		cutilSafeCall( cudaMalloc( (void**) &dScores,sizeof(float) * rowCount*colCount ));

		if (frontCount < keep)
		{
			printf(" F == TOO SMALL\n");
			calcScores_D(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
			cutilSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

			float scores[rowCount*colCount];
			cutilSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
			hToDel->clear();
			PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize,hToDel);
		}else{
			calcScores_F(dData,dFront, dScores,rowCount,colCount,frontCount);

			bool front[rowCount];
					cutilSafeCall(cudaMemcpy (front, dFront,sizeof(bool) * rowCount , cudaMemcpyDeviceToHost));

					float scores[rowCount*colCount];
					cutilSafeCall(cudaMemcpy (scores, dScores,sizeof(float) * rowCount * colCount , cudaMemcpyDeviceToHost));
					hToDel->clear();
					for(int i=0;i < rowCount;i++)
										{
											if(!front[i]){
												hToDel->push_back(i);
											}
										}
					PTeamSelectParetoSerial(frontCount < keep, front,scores, rowCount,colCount, 0, gapSize,hToDel);


			//PselectParetoSerial(frontCount < keep, front,scores, rowCount, 0, gapSize,hToDel);
		}
		cudaFree(dScores);
	}
	cudaFree(dData);
	cudaFree(dFront);
	cudaFree(dDist);

	printf(" Leave: PSel\n");
}


//
//Hey JAzz
//-- Create TEstcases for TEamSelect
//-- Integrate PointSelect
//-- reduce code
//-- Pass Tests
//-- Clean once you think it works
//
//


