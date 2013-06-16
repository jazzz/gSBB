
#ifndef _EVOCONTROLLER_
#define _EVOCONTROLLER_

#include <map>
#include <string>
#include <sys/time.h>

#include "defines.h"
#include "CudaControllerFunc.cuh"
#include <DataEngine.hpp>

using namespace std;

class EvoControllerLean
{


    public:

        string s;
        EvoControllerLean(map < string, string> &);
         ~EvoControllerLean();

         template <class keyType, class valType > void findParetoFrontDEBUG(map<keyType,vector<valType> *> *dist, set<keyType> &F, set<keyType>  &D, double epsilon,int testwhich);


        template <class keyType, class valType> void findParetoFront(map<keyType,vector<valType> *> *dist, set<keyType> &F, set<keyType> &D, double epsilon);
        template <class keyType, class valType> void findParetoFront(map<keyType,vector<valType> *> dist, set<keyType> &F, set<keyType> &D, double epsilon);
    template < class ptype > void selPareto(set < ptype > &from, map < ptype , double > &score, int nrem,set < ptype > &toDel);
    template < class ptype > void selParetoDebug(set < ptype > &from, map < ptype , double > &score, int nrem,set < ptype > &toDel);
   template < class vtype > bool dominated(vector < vtype > &, vector < vtype > &, bool &, double);
   template <class ptype, class vtype > void sharingScore(map < ptype, vector < vtype > *> &outMap,
                                        map < ptype, double> &score,
                                        set < ptype> &forThese, /* Calculate score for these members. */
                                        set < ptype> &wrtThese); /* Denominator w.r.t. these. */

   template <class ptype, class vtype > void sharingScore(map < ptype, vector < vtype > *> *outMap,
                                        map < ptype, double> &score,
                                        set < ptype> &forThese, /* Calculate score for these members. */
                                        set < ptype> &wrtThese); /* Denominator w.r.t. these. */


        void run();
        void runTests();
        set<_pointIndex> pointsToDel;
        set<_teamIndex> teamsToDel;
        set<_learnerIndex> learnersToDel;




    private:

        bool useGPU;
        DataEngine *DE;

        int _pointFrontSize;
        int _teamFrontSize;
        long _pdom;
        long _mdom;

        long time;

        void initPoints();
        void initTeams();
        void generatePrepPoints();
        void generatePoints();
        void generateTeams();

        void evaluate();

        void selectPoints();
        void selectTeams();

        void cleanup();

        void getDist(map < _pointIndex , vector < short> * > *dist);
        void EvaluateTeams();
        float evalTeam(int team_index, int point_index);

        double testTeamAgainstTrainingSet(_teamIndex tindex);
        double testTeamAgainstTestingSet_gpu(_teamIndex tindex);
        double testTeamAgainstTestingSet_cpu(_teamIndex tindex);
        void testTeam(_teamIndex tindex);
        void testTeams();
        void testTeams_gpu();
        void testTeams_cpu();


        void stats(int);
};
#endif
