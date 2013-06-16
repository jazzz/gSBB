/*
 * RandomAccessUniformLabelPointFetcher.h
 *
 *  Created on: 2013-05-29
 *      Author: jazz
 */

#ifndef RANDOMACCESSUNIFORMLABELPOINTFETCHER_H_
#define RANDOMACCESSUNIFORMLABELPOINTFETCHER_H_

#import "Dataset.hpp"

#import <vector>

class RandomAccessUniformLabelPointFetcher : PointFetcher {
public:
	RandomAccessUniformLabelPointFetcher(DataSet* DS);

	void fetchAndCopyTo(_point* p);

private:
	Dataset* DS;
	vector<_pointLabel> labels;
};

#endif /* RANDOMACCESSUNIFORMLABELPOINTFETCHER_H_ */
