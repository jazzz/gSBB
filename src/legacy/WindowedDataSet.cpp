/*
 * WindowedDataSet.cpp
 *
 *  Created on: 2013-06-05
 *      Author: jazz
 */

#include "WindowedDataSet.hpp"

WindowedDataSet::WindowedDataSet(int _d) : Dataset(_d){


}

void WindowedDataSet::setWindow(_pointIndex start, _pointIndex end)
{
	windowStart = start;
	windowEnd = end;
}

void WindowedDataSet::advanceWindow(long size)
{
	if( windowEnd+size < size_training)
	{
		setWindow(windowEnd,windowEnd+size);
	}else{
		setWindow(0,size);
	}
}

_pointIndex  WindowedDataSet::getRandomPointIndexFromWindow()
{
	return drand48()*(windowEnd-windowStart) + windowStart;
}
