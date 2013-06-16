/*
 * WindowedDataSet.hpp
 *
 *  Created on: 2013-06-05
 *      Author: jazz
 */

#ifndef WINDOWEDDATASET_HPP_
#define WINDOWEDDATASET_HPP_

#include "Dataset.hpp"

class WindowedDataSet : public Dataset {

public:

	WindowedDataSet(int _dim);
	//Window Control
	void setWindow(_pointIndex from, _pointIndex to);
	void advanceWindow(long size);

	//Access
	_pointIndex getRandomPointIndexFromWindow();

private:
	_pointIndex windowStart;
	_pointIndex windowEnd;
};

#endif /* WINDOWEDDATASET_HPP_ */
