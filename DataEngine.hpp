#ifndef DATAENGINE_HPP
#define DATAENGINE_HPP

#include "Misc.hpp"
#include "defines.h"
#include "FreeMap.hpp"
#include "Dataset.hpp"

#include "limits.h"

//#include "GpuController.cuh"

#include <set>
#include <bitset>
#include <algorithm>
#include <vector>
typedef bitset < 6 + 3 + 3 + 1 > _instruction;

using namespace std;
class DataEngine
{


    public:
        DataEngine(map < string , string > &args);
        ~DataEngine();

        bool LoadArgs(map<string, string> &args);
        void InitializeMemory();
        bool initDataSets(map < string, string> &args);
        void LoadTestRun(string pfile, string lfile, string tfile);


        // Entity Initialization

        void initPoints();
        void initTeams(long _t);

        // Entity Generation

        void generatePoints(long _t);


        void generateTeams(long _t);
        void generateTeamsNew(long _t);
        bool generateTeams(long _t, _teamIndex c);

        _learnerIndex newLearner(_learnerAction action);
        _learnerIndex newLearner(_learnerAction action, _learner* l);
        _learnerIndex newLearner(_learnerAction action, vector<_learner> &l);
        _teamIndex newTeam(long _t);

        void addLearnerToTeam(_learnerIndex li, _teamIndex ti);

        // Entity Evaluation
        void evaluate();


        _team* getPtrToTeamMatrix(int index);
        _learner* getPtrToLearnerMatrix(int index);
        _point* getPtrToPointMatrix(int index);

inline _learnerBid& getBid(_pointIndex p , _learnerIndex l)
{  return learnerBidMatrix[l* pointPool->size() + p]; }

inline _learnerAction getLearnerAction( _learnerIndex l)
{  return learnerActionMatrix[l]; }


inline _teamReward getReward(_pointIndex p, _teamIndex t)
{ return teamRewardMatrix[ t * pointPool->size() +p];}

bool mutateBid(vector<_learner>* L);

_learnerId duplicateLearner(_learnerId lid);
        void printTeamMatrix();
        void printPossibleTeamActions();
        void printLearnerMatrix();
        void printLearnerActionMatrix();
        void printLearnerReferenceCounts();
        void printPointMatrix();
        void printPointLabelMatrix();
        void printBidMatrix();
        void printTeamRewardMatrix();


        ///////////////////////////////
        // Meta Variables
        ///////////////////////////////

        int _seed;
        int statMod;
        short learnerLength;
        short pointDim;

        ///////////////////////////////
        // GP Parameters
        ///////////////////////////////


        /* Point population size. */
        int pointPopulationSize;

        /* Team population size. */
        int teamPopulationSize;

        /* Learner population size. */
        int learnerPopulationSize;

        /* Probability of learner deletion. */
        double PROB_LearnerDeletion;
        /* Probability of learner addition. */
        double PROB_LearnerAddition;
        /* Probability of action mutation. */
        double PROB_ActionMutation;
        /* Maximum team size. */
        int maxTeamSize;
        /* Number of training epochs. */
        long trainingGenerations;
        /* Point generation gap. */
        int pointGenerationGapSize;
        /* Team generation gap. */
        int teamGenerationGapSize;

        int maxProgSize;
        double PROB_BidDeletion;
        double PROB_BidAddition;
        double PROB_BidSwap;
        double PROB_BidMutate;

        int actionCount;

        ///////////////////////////////
        // Data Variables
        ///////////////////////////////
        int bytesize_TeamMatrix;
        int bytesize_TeamLengthMatrix;
        int bytesize_TeamRewardMatrix;
        int bytesize_LearnerMatrix;
        int bytesize_LearnerBidMatrix;
        int bytesize_LearnerActionMatrix;
        int bytesize_PointMatrix;
        int bytesize_PointLabelMatrix;

        ///////////////////////////////
        // Primary Data Structures
        ///////////////////////////////

        _team* teamMatrix;
        short* teamLengthMatrix;
        _teamReward* teamRewardMatrix;
        _learner* learnerMatrix;
        _learnerBid* learnerBidMatrix;
        _learnerAction* learnerActionMatrix;
        _point* pointMatrix;
        _pointLabel* pointLabelMatrix;

        Dataset* DS;

        ///////////////////////////////
        // Secondary Data Structures
        ///////////////////////////////

        FreeMap<_learnerIndex>* learnerPool;//(maxLearnerCount);
        FreeMap<_teamIndex>* teamPool;//(maxTeamCount);
        FreeMap<_pointIndex>* pointPool;//(maxPointCount);

        vector<int> newLearnerIndicies;
        vector<int> newPointIndicies;

        map< _pointIndex, long> pointAge;
        map< _teamIndex, long> teamAge;
        map<_learnerIndex, int>  learnerReferenceCounts;

        map<_pointLabelIndex, _pointLabel> labelIndexToPrintableLabel;
        map<_pointLabel,long> usedPointLabelCounts;
        set<_pointId> usedPointIds;
        map<_pointId,_pointIndex> pointIndexMap;   // Map Point ID to
        map<_pointIndex,_pointId> indexPointMap;   // Map Point ID to

    private:




};

#endif // DATAENGINE_HPP
