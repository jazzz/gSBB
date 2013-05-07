#include <iostream>
#include <sys/time.h>

#include "Misc.hpp"
#include "EvoControllerLean.hpp"
#include <stdlib.h>
using namespace std;

int main(int argc, char **argv)
{

    EvoControllerLean *model;
    map < string, string > args;
    map < string, string > :: iterator ariter;

   /////////////////////////////////////
   //  Read Arguments
   /////////////////////////////////////
    if(argc != 2)                       // Check CMDL Count
        die(__FILE__, __FUNCTION__, __LINE__, "bad arguments");
    cout << readMap(argv[1], args) << " args read" << endl;


    /////////////////////////////////////
    //  Initialize Seed
    /////////////////////////////////////
    if((ariter = args.find("seed")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg seed");
    srand48(stringToInt(ariter->second));


    /////////////////////////////////////
    //  Check EnvType
    /////////////////////////////////////
    if((ariter = args.find("envType")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg envType");

    struct timeval start, end; long mtime, seconds, useconds; gettimeofday(&start, NULL);

    if(ariter->second == "datasetEnv")
    {
        model = new EvoControllerLean(args);

        struct timeval start, end;
        long mtime, seconds, useconds;

        gettimeofday(&start, NULL);
        //=======================================

        //model->run();
        model->runTests();

        //=======================================
        gettimeofday(&end, NULL);

        seconds  = end.tv_sec  - start.tv_sec;
        useconds = end.tv_usec - start.tv_usec;

        mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;

        printf("Elapsed time: %ld milliseconds\n", mtime);
        delete model;
    }
    else
    {
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg envType");
    }


}

