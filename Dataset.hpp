#ifndef DATASET_HPP
#define DATASET_HPP

#include "defines.h"
#include <map>
#include <vector>
#include <utility>
#include <string>
#include <set>

#include <iostream>
#include "Misc.hpp"

typedef  pair<_point *, _pointLabel > PointPair;


using namespace std;

class Dataset
{
public:
    Dataset(int _dim);                                                 // Generic Object
   // Dataset(string filename, int _dim, long _size, int nfold);       // Cross Validation on monolithic DS
    void initTrainingSet(string filename,long _size);
    void initTestingSet( string filename,long _size);

    Dataset(const Dataset &);
    ~Dataset();

    _pointId getUniformTrainingPointWithLabel(_pointLabel label);
    void copyTrainingPointToMatrix(_pointId pid, _point* pointMatrix);
    void copyTestingPointToMatrix( _pointId pid, _point* pointMatrix);

    vector<_pointLabel> getLabelVector();
    vector<_pointLabel> getPrepLabelVector();

    long getPrepLabelCount(_pointLabel l);
    long getTrainingLabelCount(_pointLabel l);
    long getTestingLabelCount( _pointLabel l);
    long numLabels();


    inline long getTrainingSetSize()
    { return size_training;}

    inline long getTestingSetSize()
    { return size_testing;}

    string filename_train;
    string filename_test;
    int dim;
    long size_training;
    long size_testing;

    long windowStart;
    long windowEnd;


//private:
    //set<long> usedTrainingPointIds;
    //set<long> usedTestingPointIds;

    vector<_pointLabel> prepLabelVector;
    vector<_pointLabel> labelVector;
    map<_pointLabel,_pointLabel> labelTranslationMap;

   // map<_pointId,pair<_point *, _pointLabel>* > pointSet;

    // Remap PointSet to _point* Matrix . Index == _pointId
    _point* prepPointSet;
    _point* trainingPointSet;
    _point* testingPointSet;

    _pointLabel* prepPointSetLabels;
    _pointLabel* testingPointSetLabels;
    _pointLabel* trainingPointSetLabels;

    map<_pointLabel, vector<_pointId>* > prepLabelMap;
    map<_pointLabel, vector<_pointId>* > trainingLabelMap;
    map<_pointLabel, vector<_pointId>* > testingLabelMap;



};

#endif // DATASET_HPP
