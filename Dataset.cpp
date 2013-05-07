#include "Dataset.hpp"



Dataset::Dataset(int _d)
    :dim(_d)
{


}

void Dataset::initTrainingSet(string _filename,long _size )
{
	
	cout << "DS _SIZE:" << _size; 
    filename_train = _filename;
    size_training = _size;

    ifstream infile(_filename.c_str(), ios::in);
    if(infile == 0)
      die(__FILE__, __FUNCTION__, __LINE__, "cannot open dataset");


    // Allocate Memory
    trainingPointSet = (_point*) malloc( sizeof(_point) * dim *_size);
    trainingPointSetLabels = (_pointLabel*) malloc( sizeof(_pointLabel) *_size);

     _pointLabel tmp_label;
     int ps_index = 0;

     for(long pointId=0; pointId < _size; pointId++)
     {


         for(int d=0; d < dim ; d++)
         {
                _point tmp_p;
                 infile >> tmp_p;
                 trainingPointSet[pointId*dim+d] = tmp_p;

         }

         if(infile == 0)
            die(__FILE__, __FUNCTION__,  __LINE__, "missing pattern class");
         infile >> tmp_label;
         trainingPointSetLabels[pointId] = tmp_label;

         if(0 == trainingLabelMap.count(tmp_label)){
             labelVector.push_back(tmp_label);
             trainingLabelMap[tmp_label] = new vector<_pointId>();
         }
         trainingLabelMap[tmp_label]->push_back(pointId);

     }


}

void Dataset::initTestingSet(string _filename,long _size )
{
    filename_test= _filename;
    size_testing = _size;

	cout << "DS TEST SIZE : " << _size << endl;

    ifstream infile(_filename.c_str(), ios::in);
    if(infile == 0)
      die(__FILE__, __FUNCTION__, __LINE__, "cannot open Testing dataset");

    // Allocate Memory
    testingPointSet = (_point*) malloc( sizeof(_point) * dim *_size);
    testingPointSetLabels = (_pointLabel*) malloc( sizeof(_pointLabel) *_size);

     _pointLabel tmp_label;

     for(long pointId=0; pointId < _size; pointId++)
     {

         for(int d=0; d < dim ; d++)
         {
                 infile >> testingPointSet[pointId*dim+d];
         }

         if(infile == 0)
            die(__FILE__, __FUNCTION__,  __LINE__, "missing pattern class");
         infile >> tmp_label;
         testingPointSetLabels[pointId] = tmp_label;


         if(0 == testingLabelMap.count(tmp_label)){
             testingLabelMap[tmp_label] = new vector<_pointId>();
         }
         testingLabelMap[tmp_label]->push_back(pointId);

     }
}


Dataset::~Dataset()
{

//    map<long,pair<_point*, _pointLabel>* >::iterator it, itend;
//    it = pointSet.begin();
//    itend = pointSet.end();

//    for(;it != itend; it++)
//    {
//        pair<_point*, _pointLabel> *p = (*it).second;
//        delete p->first;
//        delete p;
//    }
        cout << "DS  DETOR" << endl;

      map<_pointLabel, vector<_pointId>*>::iterator it;
      for(it= trainingLabelMap.begin(); it != trainingLabelMap.end(); it++)
      {
        delete (*it).second;
      }

      for(it= testingLabelMap.begin(); it != testingLabelMap.end(); it++)
      {
         delete (*it).second;
      }
      free(trainingPointSet);
      free(testingPointSet);
      free(trainingPointSetLabels);
      free(testingPointSetLabels);

}

Dataset::Dataset(const Dataset& D )
{

}

long Dataset::numLabels()
{

    return labelVector.size();

}

long Dataset::getTrainingLabelCount(_pointLabel l)
{
//    cout << " IN GTLC: " << l <<  endl;
//    cout << &trainingLabelMap << endl;
//    cout << trainingLabelMap.size() << endl;
//
//
//
//     map<_pointLabel, vector<_pointId>*>::iterator it;
//     for(it= trainingLabelMap.begin(); it != trainingLabelMap.end(); it++)
//      {
//        cout << "--$ " << endl;
//        cout << "-->" << (*it).first << "    " << (*it).second->size() << endl;
//      }
//
    return trainingLabelMap[l]->size();
}
long Dataset::getTestingLabelCount(_pointLabel l)
{
    return testingLabelMap[l]->size();
}

vector<_pointLabel> Dataset::getLabelVector()
{
        return labelVector;
}


_pointId Dataset::getUniformTrainingPointWithLabel(_pointLabel label)
{
 //  cout << "LabeL:" << label << "NPIndex:" << drand48() * trainingLabelMap[label]->size() << " NPId" <<  (*trainingLabelMap[label])[drand48() * trainingLabelMap[label]->size() ] << endl;
     return (*trainingLabelMap[label])[drand48() * trainingLabelMap[label]->size() ];

}

void Dataset::copyTrainingPointToMatrix(_pointId pid, _point* pointMatrix)
{
    for(int i=dim-1; i>=0; i--)
    {

       pointMatrix[i] = trainingPointSet[pid*dim+i];
    }
}
void Dataset::copyTestingPointToMatrix(_pointId pid, _point* pointMatrix)
{
    for(int i=dim-1; i>=0; i--)
    {
       pointMatrix[i] = testingPointSet[pid*dim+i];
    }
}
