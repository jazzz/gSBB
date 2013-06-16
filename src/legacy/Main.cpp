

#include "DataPartition.hpp"
#include "../Misc.hpp"
#include <fstream>

using namespace std;

int main2(){

	int dim = 9;
	int pCount = 40;
	DataPartition* DP = new DataPartition(pCount, dim);

	ifstream infile("../../local_datasets/FORMATED_shuttle.data", ios::in);
	if(infile == 0){
	   	die(__FILE__, __FUNCTION__, __LINE__, "cannot open dataset");
    }

	_point point[dim];
	_pointLabel label;
	for(int pid =0; pid <  pCount; pid++)
	{

		for(int d=0; d < dim ; d++){ point[d] =0;}
		for(int d=0; d < dim ; d++)
		{
			infile >> point[d];
		}
		if(infile == 0)
		  	die(__FILE__, __FUNCTION__,  __LINE__, "missing pattern class");
		infile >> label;

		DP->addPoint(point,label);
	}

	return 1;
}
