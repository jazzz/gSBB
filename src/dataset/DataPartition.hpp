/*
 * DataPartition.hpp
 *
 *  Created on: 2013-06-16
 *      Author: jazz
 */

#ifndef DATAPARTITION_HPP_
#define DATAPARTITION_HPP_

#include <defines.h>
#include <cstdlib>
#include <cassert>


class DataPartition {
public:
	DataPartition(int maxSize, int dim);
	~DataPartition();

	// Initialization / Management
	void indexPoints();
	void addPoint(_point*,_pointLabel);

	inline void markdirty(){
		this->_isIndexed = false;
	}

	// Query State
	inline bool isIndexed(){
		return _isIndexed;
	}
	inline bool hasRoom(){
		return (_size < _capacity) ? true : false;
	}

	inline int size(){
		return _size;
	}


	// Fetch points
	_pointId getUniformPointWithLabel(_pointLabel label);
	_pointId getRandomPointWithLabel(_pointLabel label);

	// helpers
	void copyPointToMatrix(_pointId pid, _point* pointMatrix);


	//Variables
private:
	bool _isIndexed;

	int _capacity;
	int _size;
	int _dim;

	_point* _pointSet;
	_pointLabel* _pointLabelSet;

};

#endif /* DATAPARTITION_HPP_ */
