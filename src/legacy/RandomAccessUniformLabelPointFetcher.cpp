/*
 * RandomAccessUniformLabelPointFetcher.cpp
 *
 *  Created on: 2013-05-29
 *      Author: jazz
 */

#include "RandomAccessUniformLabelPointFetcher.hpp"

RandomAccessUniformLabelPointFetcher::RandomAccessUniformLabelPointFetcher(Dataset* ds)
{
	DS = ds;
	labels = DS->getLabelVector();
}

RandomAccessUniformLabelPointFetcher::fetchAndCopyTo(_point* p)
{
	v
	int __i =0;
	do
	{
		label = labels[drand48() * labels.size()] ;
		if( __i++ > 100 ) {cout << "DEAD LOCK" << endl;}
	} while (usedPointLabelCounts[label] == DS->getTrainingLabelCount(label));

	_pointId pid = DS->getUniformTrainingPointWithLabel(label);

}

RandomAccessUniformLabelPointFetcher::labelVector(){
	return DS->labelVector;
}
