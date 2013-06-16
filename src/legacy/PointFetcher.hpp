/*
 * PointFetcher.hpp
 *
 *  Created on: 2013-05-29
 *      Author: jazz
 */

#ifndef POINTFETCHER_HPP_
#define POINTFETCHER_HPP_

#include "defines.h"

#include <vector>

class PointFetcher {
public:
	virtual bool fetchAndCopyTo(_point* p) = 0;
};



#endif /* POINTFETCHER_HPP_ */
