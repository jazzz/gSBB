
#ifndef _GPUTESTMAIN_
#define _GPUTESTMAIN_

#include <map>
#include <string>
#include <sys/time.h>

extern "C" void GPU_go();

#define DIFF(A,B){1000000*(B.tv_sec - A.tv_sec) + B.tv_usec - A.tv_usec };

long Diff(timeval tv_start, timeval tv_end){
   return 1000000*(tv_end.tv_sec - tv_start.tv_sec) + tv_end.tv_usec - tv_start.tv_usec;
}


int main(char** argv, int argc)
{
    struct timeval tv_start;
    struct timeval tv_end;
    struct timezone tz;
    long timerTest = 0;
//    cout << "" << endl;
//    gettimeofday(&tv_start, &tz);
//
	GPU_Go();

//	gettimeofday(&tv_end, &tz);
//    timerTest = Diff(tv_start, tv_end);
//
//    printf("TESTTIME %lu\n", timerTest);
}

#endif
