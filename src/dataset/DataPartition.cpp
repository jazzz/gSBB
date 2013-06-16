/*
 * DataPartition.cpp
 *
 *  Created on: 2013-06-16
 *      Author: jazz
 */

#include "DataPartition.hpp"

DataPartition::DataPartition(int maxSize ,int dim)
{
	this->_capacity = maxSize;
	this->_size = 0;
	this->_dim = dim;

	this->_pointSet = (_point*) malloc( sizeof(_point) * _dim * _capacity);
	this->_pointLabelSet = (_pointLabel*) malloc( sizeof(_pointLabel) * _capacity);

}

DataPartition::~DataPartition()
{

}

/////////////////////////////////
// Initialization
/////////////////////////////////

void DataPartition::indexPoints()
{
//TODO
}
void DataPartition::addPoint(_point* newPoint, _pointLabel newLabel)
{
	assert(this->hasRoom());
	for(int i=0; i < _dim;i++) _pointSet[_size*_dim+i] = newPoint[i];
	_pointLabelSet[_size] = newLabel;
}


// Fetch points
_pointId DataPartition::getUniformPointWithLabel(_pointLabel label){

}
_pointId DataPartition::getRandomPointWithLabel(_pointLabel label){

}

// helpers
void DataPartition::copyPointToMatrix(_pointId pid, _point* pointMatrix){

}

