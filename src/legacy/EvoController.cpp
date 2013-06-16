
 #define CPU_TEAM_EVAL
#include "EvoController.hpp"
#include "GpuController_simple.cuh"
#include "GpuTestController.cuh"
#include "GpuController_Selection.cuh"

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#ifdef SELECTIVE_EVAL
void SelectiveLearnerEval(int, _learner*, int, _point*, _learnerBid*, int, int* ,int ,int*);
#endif

#define DIFF(A,B){1000000*(B.tv_sec - A.tv_sec) + B.tv_usec - A.tv_usec };

/////////////////////////////////
// Constructors
/////////////////////////////////
bool GPU_SELECT_TEAMS = false;
bool GPU_SELECT_POINTS = true;

EvoController::EvoController(map < string, string> &args)
{
   DE = new DataEngine(args);
   useGPU = true;
  // InitGPU(DE->learnerMatrix, DE->learnerBidMatrix, DE->pointMatrix, useGPU);
   InitGPU(DE->learnerPopulationSize,DE->pointPopulationSize, DE->maxProgSize,DE->pointDim, useGPU);
// PushConstants(DE->maxProgSize, DE->learnerLength,DE->pointDim );

   cout << "#USE GPU " << useGPU << endl;

   _teamFrontSize = 0;
   _pointFrontSize =0;
   _pdom = 0;
   _mdom = 0;
}

EvoController::~EvoController()
{
    GpuCleanup();
    //InitializeGPUSelection();
    delete (DE);
}

long Diff(timeval tv_start, timeval tv_end){
   return 1000000*(tv_end.tv_sec - tv_start.tv_sec) + tv_end.tv_usec - tv_start.tv_usec;
}

void EvalTeams2( _learnerBid* bids )
{	
	


}

/////////////////////////////////
// Main GP Loop
/////////////////////////////////
void EvoController::runTests()
{


	 typename map<int, vector< int>* > ::iterator veiter1, veiter2, veiterbegin, veiterend;

	    vector<int> *v1,*v2;

	    bool dom;
	    bool equal;

	    int i,j;

	    map<int,vector<int>* > dist;
	    vector<int> *v;

	    v = new vector<int>;
	    v->push_back( 0 ); v->push_back( 1 ); v->push_back( 1 ); v->push_back( 0 );
	    dist[0] = v;
	    v = new vector<int>;
	    v->push_back( 1 ); v->push_back( 0 ); v->push_back( 1 ); v->push_back( 0 );
	   	dist[1] = v;
	   	v = new vector<int>;
	    v->push_back( 1 ); v->push_back( 0 ); v->push_back( 0 ); v->push_back( 0 );
	   	dist[2] = v;
	   	v = new vector<int>;
	    v->push_back( 0 ); v->push_back( 1 ); v->push_back( 0 ); v->push_back( 0 );
	   	dist[3] = v;




	    for(i = 0, veiterbegin = dist.begin(), veiter1 = dist.begin(), veiterend = dist.end();
	        veiter1 != veiterend; veiter1++, i++)
	    {
	        dom = 0;
	        v1 = veiter1->second;

	        for(j = 0, veiter2 = veiterbegin; dom == 0 && veiter2 != veiterend; veiter2++, j++)
	        {
	            v2 = veiter2->second;
	            dom = dominated(*v1, *v2, equal, 0.001);
	            cout << " " << dom;
	        }
	        cout << endl;
	    }

if(true)
	return;

	cout << "TEST RUN" <<endl;
	DE->LoadTestRun("points.txt", "learners.txt", "teams.txt");
	
	cout << "TEAM" << DE->teamPool->size() << endl;

	for(int i=0; i < 5; i++)
	{
		cout << "LID: " << i << " A: " << DE->learnerActionMatrix[i] << endl;
	}
//	DE->printPointMatrix();
//	DE->printTeamMatrix();
//	DE->printLearnerMatrix();

	//evaluate();
	


	int dim = DE->DS->dim;
	int pointCount = DE->pointPopulationSize;
	int learnerCount = DE->learnerPool->maxUsedIndex();
	_learnerBid LBA[pointCount*learnerCount];


	if(useGPU == false)
	{
		for(int LID = 0;LID<learnerCount; LID++)
		{
			for(int PID=0; PID< DE->pointPopulationSize;PID++)
			{
				int feats = DE->DS->dim;

				_learnerBid QQ = -1;
				cLearnerEvalSingle(&DE->learnerMatrix[LID*DE->learnerLength],QQ,&DE->pointMatrix[PID*feats],DE->learnerLength,feats);
				cout << "============" << QQ << endl;
				DE->learnerBidMatrix[LID*pointCount+PID] = QQ;

			}
		}
	}
	else {
        EvaluateLearners(DE->learnerPool->maxUsedIndex()+1, DE->learnerMatrix, DE->learnerBidMatrix,  DE->pointPool->size(), DE->pointMatrix);
	}


	cout << " BABABABABAB" << endl;
	for(int LID = 0;LID<7; LID++)
	{
		for(int PID=0; PID< DE->pointPopulationSize;PID++)
		{
			cout << " " <<DE->learnerBidMatrix[LID*pointCount+PID];

		}
		cout << endl;
	}



	//for(int i = 0;i< DE->learnerPopulationSize; i++)
	for(int i = 0;i< 7; i++)
	{
		for(int j=0; j< DE->pointPopulationSize;j++)
		{

			printf("\nL%d: ", i);
			for(int q =1;q <=DE->learnerMatrix[i*DE->learnerLength+0];q++)
			{
				printf(" %hd", DE->learnerMatrix[i*DE->learnerLength+q]);
			}

			printf("\nP%d: ",j);
			for(int w=0; w < dim;w++)
			{
				printf(" %0.1f",DE->pointMatrix[j*dim+w]);
			}
			printf("\n$%0.3f", DE->learnerBidMatrix[i*pointCount+j]);

		}
	}

	cout << "===========\n===============\n==============\n============\n" <<endl;

	//DE->printPointMatrix();

	DE->printTeamRewardMatrix();

	int LID = 6;
	int PID = 8;
	_learnerBid LB;

	int feats = DE->DS->dim;
	cLearnerEvalSingle(&DE->learnerMatrix[LID*DE->learnerLength],LB,&DE->pointMatrix[PID*feats],DE->learnerLength,feats);
	printf(" %f", LB);

	for(int i=0; i < 5; i++)
	{
		cout << "LID: " << i << " A: " << DE->learnerActionMatrix[i] << endl;
	}


    EvaluateTeams();
	DE->printBidMatrix();
	cout << "REWARDS" <<endl;
	DE->printTeamRewardMatrix();
	testTeams();

}

void EvoController::run()
{


    cout << " Start Run" << endl;

    initPoints();
    initTeams();

	evaluate();
    int injectTime = -2;
    time =1;


    struct timeval tv_start;
    struct timeval tv_end;
    struct timezone tz;
    long evalTimerSum =0;
    long timerPSelect = 0;
    long timerTSelect = 0;
    long timerPGen = 0;
    long timerTGen = 0;
    long timerEval = 0;
    long timerCleanup = 0;
    long timerTest = 0;

int sumEvalCount = 0;

    for(; time < DE->trainingGenerations; time++)
    {
    	  cout << " Select: " << time  << endl;

        gettimeofday(&tv_start, &tz);
        selectPoints();
        gettimeofday(&tv_end, &tz);
        timerPSelect = Diff(tv_start,tv_end);


        gettimeofday(&tv_start, &tz);
        selectTeams();
        gettimeofday(&tv_end, &tz);
        timerTSelect = Diff(tv_start,tv_end);

     //   cout << "POSSIBLE" << endl;
     //   DE->printPossibleTeamActions();
        gettimeofday(&tv_start, &tz);
        cleanup();
        gettimeofday(&tv_end, &tz);
        timerCleanup = Diff(tv_start,tv_end);
        cout << " GENERATION: " << time  << endl;


        gettimeofday(&tv_start, &tz);
        generatePoints();
        gettimeofday(&tv_end, &tz);
        timerPGen = Diff(tv_start,tv_end);

        gettimeofday(&tv_start, &tz);
        generateTeams();
        gettimeofday(&tv_end, &tz);
        timerTGen = Diff(tv_start,tv_end);


        if (time == injectTime)
        {
            cout << "IMPORT_DATA" <<endl;
            DE->LoadTestRun("points.txt", "learners.txt", "teams.txt");
        }


        if (time == injectTime+1)
        {
            cout << "IMPORT_POINTS" <<endl;
            DE->LoadTestRun("points.txt", "", "");
        }

        cout << " EVAL: " << time  << endl;

        gettimeofday(&tv_start, &tz);
        evaluate();
        gettimeofday(&tv_end, &tz);
        timerEval = Diff(tv_start, tv_end);

        if (time == injectTime)
        {
            cout << "=============================\n========================== INJECTED\n=========================" <<endl;
            DE->printBidMatrix();
            DE->printTeamRewardMatrix();
        }

        if(time%DE->statMod == 0)
            stats(time);
        if (time == injectTime)
        {

            cout << "Inject" <<endl;
            DE->printTeamMatrix();
            DE->printLearnerMatrix();
            testTeams();
            cout << "IMPORT_DATA_END" <<endl;


        }

        if (time == injectTime+1)
        {

            cout << "Inject+1" <<endl;
            DE->printTeamMatrix();
            DE->printLearnerMatrix();
            testTeams();
            cout << "IMPORT_POINTS_END" <<endl;
        }


        printf("TIMESTAT %lu %lu  %lu  %lu  %lu  %lu\n", timerPSelect , timerTSelect, timerCleanup,timerPGen , timerTGen, timerEval);

    }
 //   DE->printTeamMatrix();
  //     DE->printBidMatrix();
    //  generateTeams();

    cout << "FINAL TEST NOT IMPLEMENTED" << endl;
    gettimeofday(&tv_start, &tz);
  testTeams();
    gettimeofday(&tv_end, &tz);
    timerTest = Diff(tv_start, tv_end);

    printf("TESTTIME %lu\n", timerTest);


    DE->printPossibleTeamActions();
}


/////////////////////////////////
// GP Functions
/////////////////////////////////

void EvoController::initPoints()
{
   DE->initPoints();
}

void EvoController::initTeams()
{
   DE->initTeams(time);
}


void EvoController::generatePoints()
{
   DE->generatePoints(time);
}

void EvoController::generateTeams()
{
   DE->generateTeams(time);
}

void EvoController::evaluate()
{

        timespec evalStartTime;
        timespec evalEndTime;

        clock_gettime(CLOCK_MONOTONIC, &evalStartTime);
    //DE->printPointMatrix();
   // DE->printLearnerMatrix();

    if (useGPU)
    {

    	#ifndef SELECTIVE_EVAL
        EvaluateLearners(DE->learnerPool->maxUsedIndex()+1, DE->learnerMatrix, DE->learnerBidMatrix,  DE->pointPool->size(), DE->pointMatrix);
      //  EvaluateLearners(8, DE->learnerMatrix, DE->learnerBidMatrix, 3, DE->pointMatrix);
		#endif
		#ifdef SELECTIVE_EVAL
        	SelectiveLearnerEval(DE->learnerPool->maxUsedIndex()+1, DE->learnerMatrix, DE->pointPool->size(), DE->pointMatrix, DE->learnerBidMatrix, DE->newLearnerIndicies.size(), &DE->newLearnerIndicies[0], DE->newPointIndicies.size(), &DE->newPointIndicies[0]);
		#endif

    }
    else
    {

        set<_pointIndex>::iterator pit, pitend;
        set<_learnerIndex>::iterator lit, litend;



        for(lit= DE->learnerPool->begin(), litend = DE->learnerPool->end(); lit != litend; lit++)
        {
      // for( int lit =0 ; lit < DE->learnerPool->maxUsedIndex()+1; lit++)
      // {
       // cout << (*lit) << endl;
         //   int qw = 0;
            for(pit= DE->pointPool->begin(), pitend = DE->pointPool->end(); pit != pitend; pit++)
            {
            //    if (qw++ >1) { break;}
                _learner* L = DE->getPtrToLearnerMatrix((*lit));
                _point* P = DE->getPtrToPointMatrix((*pit));

                cLearnerEvalSingle(L, DE->getBid((*pit), (*lit)), P, DE->learnerLength, DE->pointDim );
            }


        }
    }


    clock_gettime(CLOCK_MONOTONIC, &evalEndTime);
   // DE->printBidMatrix();
//     DE->printLearnerMatrix();
    // DE->printBidMatrix();

    //    for(int _i=0; _i< (DE->learnerPool->maxUsedIndex()+1)*DE->pointPool->size(); _i++)
    //    {
    //        DE->learnerBidMatrix[_i] =  floor(DE->learnerBidMatrix[_i]* 1000.0f) / 1000.0f;
    //  }
   // DE->printPointMatrix();
   // DE->printBidMatrix();
    //  DE->printLearnerMatrix();
    EvaluateTeams();

    //DE->printLearnerMatrix();
   // DE->printLearnerMatrix();
   // DE->printBidMatrix();

    long time_elapsed = (evalEndTime.tv_sec - evalStartTime.tv_sec) * 1000000000 + (evalEndTime.tv_nsec - evalStartTime.tv_nsec);
    cout <<" LearnerEvalTime: " << time_elapsed << endl;

    DE->newLearnerIndicies.clear();
    DE->newPointIndicies.clear();
}

void EvoController::EvaluateTeams()
{
    int val = 0;

//    cout << " ===================\n=========\n==========\n==========\n EVAL TEAMS: " << DE->teamPool->size() << "   " << DE->pointPool->size() << endl;
//	cout << DE->teamPool->size() << "   " << DE->pointPool->size() << endl;

    for (int team_index=0; team_index < DE->teamPool->size(); team_index++)
    {
            for (int point_index =0; point_index < DE->pointPool->size();point_index++)
            {
               // if (team_index == 3)
               // {

               // DE->teamRewardMatrix[point_index + team_index*DE->pointPool->size()] = 6 ;
              //  }
			  //cout << te;
                DE->teamRewardMatrix[point_index + team_index*DE->pointPool->size()] = evalTeam(team_index, point_index) ;

            }
    }
}

float truncate(float n , int p)
{
	return floor(pow(10, p) * n) / pow(10, p);
}
float EvoController::evalTeam(int team_index, int point_index)
{

//    cout << "Team" << team_index ;
    _team* Team = &DE->teamMatrix[team_index*DE->maxTeamSize];
    int j=0;
  //  while(Team[j] != -1 && j < DE->maxTeamSize)
  //  {
  //      cout << "   " << Team[j++];
  //  } cout <<  " | Point " << point_index <<  endl;

    float eps = 0.00000001;

   _learnerBid bestBid = -9999;
   _learnerIndex bestBidder = 0;

   //cout << "T("<<team_index << ": " << point_index <<") ";
   vector<_learnerIndex> vec;
    int i =0;
    while(Team[i] != -1 && i < DE->maxTeamSize)
    {


        _learnerIndex lindex = Team[i];
        _learnerBid currBid = DE->learnerBidMatrix[lindex * DE->pointPool->size() + point_index];
 //       cout << lindex<<"|" << currBid << " ";
        if ( currBid > bestBid  ) //|| ( currBid == bestBid && lindex < bestBidder))
        {
            bestBid = currBid;
            bestBidder = lindex;
            vec.clear();
            vec.push_back(lindex);
        }
        else if( currBid == bestBid)                                   // TODO
        {
        	vec.push_back(lindex);

        }

        i++;
    }

//    if(vec.size() > 1)
//    {
//    	bestBidder = vec[rand() % vec.size()];
//    }
	
    _learnerAction action = DE->learnerActionMatrix[bestBidder];



//    cout << "--  TopBidder: " << bestBidder ;
//    cout << " BestBid: "<< bestBid << "  BidAction: ";
//    cout << action << " TrueAction: " << DE->pointLabelMatrix[point_index] ;
//    cout << " Reward: " << (DE->pointLabelMatrix[point_index] == action) << endl;
//


    if (DE->pointLabelMatrix[point_index] == action)
    {
        return 1;
    }else{
        return 0;
    }

}
void EvoController::getDist(map < _pointIndex, vector < short> * > *dist)
{
    _learnerBid outcomes[DE->teamPool->size()+1];
    double out;
    vector < short > *distvec;

   int i,j;

    set<_pointIndex>::iterator pit;

    for (pit = DE->pointPool->begin(); pit != DE->pointPool->end(); pit++)
    {
        _pointIndex pindex = (*pit);
        set<_teamIndex>::iterator tit;
        int count =0;
       // printf("\n$$ %d) ", pindex);
        for (tit = DE->teamPool->begin(); tit != DE->teamPool->end(); tit++)
        {
            _teamIndex tindex = (*tit);

         //   printf("  %d",DE->getReward(pindex,tindex));
            outcomes[count++] = DE->getReward(pindex,tindex);

        }
       // printf("\n");

        distvec = new vector < short>;
        (*dist)[pindex] = distvec;


        int size = DE->teamPool->size();


        for(i = 0; i < size; i++){
            for(j = 0; j < size; j++)
            {
                /* See if outcome i > j. */
                //if(outcomes[i] > outcomes[j] && isEqual(outcomes[i], outcomes[j]) == 0)
           // 	printf(" %d", (outcomes[i] > outcomes[j]));
                if(outcomes[i] > outcomes[j] )
                {
                    distvec->push_back(1);
                }else{
                    distvec->push_back(0);
                }
            }
           // printf("\n");
      }


    }

}

template < class vtype > bool EvoController::dominated(vector < vtype > &is, /* Is this vector */
                         vector < vtype > &by, /* dominated by this vector. */
                         bool &equal, /* Whether or not they are equal. */
                         double epsilon)
{
  /* Assume higher outcomes better. */

  if(is.size() != by.size())
    die(__FILE__, __FUNCTION__, __LINE__, "outcome vectors do not match");

  typename vector < vtype > :: iterator isIter, byIter, endIter;

  /* Initially assume `is' is equal to 'by'. */
  equal = 1;

//  cout << " IS: ";
//  for(isIter = is.begin(), endIter = is.end(); isIter != endIter; isIter++)
//    {
//        cout <<"  " << (*isIter);
//    }
//  cout << endl << " BY: ";
//  for(isIter = by.begin(), endIter = by.end(); isIter != endIter; isIter++)
//    {
//        cout <<"  " << (*isIter);
//    }
// cout << endl << " || ";

  for(isIter = is.begin(), byIter = by.begin(), endIter = is.end();
      isIter != endIter; isIter++, byIter++)
    {
      /* They are equal in this dimension. */
      if(fabs(*isIter - *byIter) < epsilon) continue;

      /* At this point know they are not equal. */
      equal = 0;

      /* Not dominated since 'is' greater than 'by' in this dimension. */

//     if(*isIter > *byIter) cout << " Not Dom Not Equal" << endl;
      if(*isIter > *byIter) return 0;
    }

  /* At this point, 'is' is never larger than 'by' (otherwise would have returned
     already), so if they are not equal 'by' must be larger than 'is' on some
     dimension and therefore dominant. */

//  if(equal) cout << " IS Equal" << endl;
//  else cout << " IS Dom " << endl;
  return equal == 0;
}

template < class ptype > void EvoController::selPareto(set < ptype  > &from,
                         map < ptype , double > &score,
                         int nrem,
                         set < ptype > &toDel)
{
  typename set < ptype > :: iterator ptiter, ptiterend;
  vector < pair < ptype, double>  > ptvec;
  typename vector < pair<ptype, double > > :: iterator ptv_it;


  for(ptiter = from.begin(), ptiterend = from.end();
      ptiter != ptiterend; ptiter++)
    {
      ptvec.push_back( pair<ptype,double>( *ptiter, (score.find(*ptiter))->second) );
      //(*ptiter)->key((score.find(*ptiter))->second);
      //(*ptiter).key((score.find(*ptiter))->second);
      //ptvec.push_back(*ptiter);
    }
//  cout << "PTVEC Init: " ;
//  for(ptv_it = ptvec.begin(); ptv_it != ptvec.end(); ptv_it++)
//  {
//        cout << "   " << (*ptv_it).first << ":" << (*ptv_it).second;
//  }
//  cout << endl;

 // partial_sort(ptvec.begin(), ptvec.begin() + nrem, ptvec.end(), lessThan < ptype > () );
  partial_sort(ptvec.begin(), ptvec.begin() + nrem, ptvec.end(),compare_pair_second<std::less>() );
// cout << "PTVEC Sort: " ;
//  for(ptv_it = ptvec.begin(); ptv_it != ptvec.end(); ptv_it++)
//  {
//        cout << "   " << (*ptv_it).first << ":" << (*ptv_it).second;
//  }
//  cout << endl;

//  toDel.insert(ptvec.begin(), ptvec.begin() + nrem);
   for(ptv_it = ptvec.begin(); ptv_it != ptvec.begin() + nrem; ptv_it++)
  {
toDel.insert( (*ptv_it).first);
  }

//  cout << "PTVEC FirstN: " ;
//  for(ptv_it = ptvec.begin(); ptv_it != ptvec.begin()+nrem; ptv_it++)
//  {
//        cout << "   " << (*ptv_it).first << ":" << (*ptv_it).second;
//  }
//  cout << endl;

#ifdef MYDEBUG
  cout << "scm::selPareto " << nrem << "/" << from.size();
  for(int i = 0; i < ptvec.size(); i++)
    cout << " " << *ptvec[i] << "/" << (score.find(ptvec[i]))->second;
  cout << endl;

  cout << "scm::selPareto deleting";
  for(ptiter = toDel.begin(); ptiter != toDel.end(); ptiter++)
    cout << " " << **ptiter;
  cout << endl;
#endif
}


template <class keyType, class valType > void EvoController::findParetoFront(map<keyType,vector<valType> *> dist, set<keyType> &F, set<keyType>  &D, double epsilon)
{
    typename map<keyType, vector< valType>* > ::iterator veiter1, veiter2, veiterbegin, veiterend;

    vector<valType> *v1,*v2;

    bool dom;
    bool equal;

    int i,j;

//    for(i = 0, veiterbegin = dist.begin(), veiter1 = dist.begin(), veiterend = dist.end();
//        veiter1 != veiterend; veiter1++, i++)
//    {
//        cout << i << ")\t[";
//        for (int _j =0; _j < (*veiter1).second->size(); _j++)
//        {
//                cout << "   " << (*(*veiter1).second)[_j];
//        }
//        cout << "]"<<endl;
//    }


    for(i = 0, veiterbegin = dist.begin(), veiter1 = dist.begin(), veiterend = dist.end();
        veiter1 != veiterend; veiter1++, i++)
    {
        dom = 0;
        v1 = veiter1->second;

        for(j = 0, veiter2 = veiterbegin; dom == 0 && veiter2 != veiterend; veiter2++, j++)
        {
            v2 = veiter2->second;
            dom = dominated(*v1, *v2, equal, epsilon);

            /* Also dominated if equal to a previously processed item. */
            if(j < i && equal == 1)
            {
                dom = 1;
            }
#ifdef MYDEBUG
            if(dom == 1)
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " yesdom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
            else
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " nondom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
#endif
        }

        if(dom == 0)
           	F.insert(veiter1->first);
        else
        	D.insert(veiter1->first);
    }

}


template <class keyType, class valType > void EvoController::findParetoFront(map<keyType,vector<valType> *> *dist, set<keyType> &F, set<keyType>  &D, double epsilon)
{
    typename map<keyType, vector< valType>* > ::iterator veiter1, veiter2, veiterbegin, veiterend;

    vector<valType> *v1,*v2;

    bool dom;
    bool equal;

    int i,j;

//    for(i = 0, veiterbegin = dist.begin(), veiter1 = dist.begin(), veiterend = dist.end();
//        veiter1 != veiterend; veiter1++, i++)
//    {
//        cout << i << ")\t[";
//        for (int _j =0; _j < (*veiter1).second->size(); _j++)
//        {
//                cout << "   " << (*(*veiter1).second)[_j];
//        }
//        cout << "]"<<endl;
//    }


    for(i = 0, veiterbegin = dist->begin(), veiter1 = dist->begin(), veiterend = dist->end();
        veiter1 != veiterend; veiter1++, i++)
    {
        dom = 0;
        v1 = veiter1->second;
   //     printf("## %d=  ", i);
        for(j = 0, veiter2 = veiterbegin; dom == 0 && veiter2 != veiterend; veiter2++, j++)
        {
            v2 = veiter2->second;
            dom = dominated(*v1, *v2, equal, epsilon);

  //          printf(" %d", (dom) ? 2 : (j < i && equal == 1) ? 1 : 0 );
   //         if( dom ==1)
   //             cout << "   (" << j <<" dom "<< i << ")";
            /* Also dominated if equal to a previously processed item. */
            if(j < i && equal == 1)
            {
                dom = 1;
     //           cout << "   (" << i <<" equal "<< j << ")";
            }

#ifdef MYDEBUG
            if(dom == 1)
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " yesdom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
            else
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " nondom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
#endif
        }
     //   printf("\n");

        if(dom == 0)
        {
      //      cout << "   (" <<i << " champ)";
      F.insert(veiter1->first);
        }else
      D.insert(veiter1->first);
    }
//    set< _pointIndex> :: iterator poiter, poiterend;
//    cout << "scm::selPoints F1";
//    for(poiter = F.begin(); poiter != F.end(); poiter++)
//      cout << " " << (*poiter);
//    cout << endl;
//
//    cout << "scm::selPoints D2";
//    for(poiter = D.begin(); poiter != D.end(); poiter++)
//      cout << " " << (*poiter);
//    cout << endl;
}
template <class keyType, class valType > void EvoController::findParetoFrontDEBUG(map<keyType,vector<valType> *> *dist, set<keyType> &F, set<keyType>  &D, double epsilon,int testwhich)
{
    typename map<keyType, vector< valType>* > ::iterator veiter1, veiter2, veiterbegin, veiterend;

    vector<valType> *v1,*v2;

    bool dom;
    bool equal;

    int i,j;

 //   printf("\n@#$%\n");
    for(i = 0, veiterbegin = dist->begin(), veiter1 = dist->begin(), veiterend = dist->end();
        veiter1 != veiterend; veiter1++, i++)
    {
        dom = 0;
        v1 = veiter1->second;
        for(j = 0, veiter2 = veiterbegin; dom == 0 && veiter2 != veiterend; veiter2++, j++)
        {
            v2 = veiter2->second;
            dom = dominated(*v1, *v2, equal, epsilon);
//
//            if(testwhich == 0)
//            {
//            	printf(" %d", (dom) ? 1 : 0 );
//            }
//            if(testwhich ==1)
//            {
//            	printf(" %d", (equal) ? 1 : 0 );
//            }


            /* Also dominated if equal to a previously processed item. */
            if(j < i && equal == 1)
            {
            //    dom = 1;  TODO: Dont forget ot uncomment
            }

#ifdef MYDEBUG
            if(dom == 1)
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " yesdom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
            else
                cout << "scm::findParetoFront " << veiter1->first->id() << " vecs" << vecToStr(*v1) << " nondom " << veiter2->first->id() << " vecs" << vecToStr(*v2) << endl;
#endif
        }
        printf("\n");

        if(dom == 0)
        {
      //      cout << "   (" <<i << " champ)";
      F.insert(veiter1->first);
        }else
      D.insert(veiter1->first);
    }
//    set< _pointIndex> :: iterator poiter, poiterend;
//    cout << "scm::selPoints F1";
//    for(poiter = F.begin(); poiter != F.end(); poiter++)
//      cout << " " << (*poiter);
//    cout << endl;
//
//    cout << "scm::selPoints D2";
//    for(poiter = D.begin(); poiter != D.end(); poiter++)
//      cout << " " << (*poiter);
//    cout << endl;
}

 template <class ptype, class vtype > void EvoController::sharingScore(map < ptype , vector < vtype > * > &outMap,
                                    map < ptype , double > &score,
                                    set < ptype  > &forThese, /* Calculate score for these members. */
                                    set < ptype  > &wrtThese) /* Denominator w.r.t. these. */
    {
      if(score.empty() == 0)
        die(__FILE__, __FUNCTION__, __LINE__, "score map should be empty");

      typename set < ptype > :: iterator ptiter, ptiterend;

      /* Denominator in each dimension. */
      vector < vtype > nd;

      vector < vtype > *outvec;
      double sc;
      int i;

      int outvecsize = (outMap.begin()->second)->size();

      /* Initialize to 1 so that do not divide by zero. */
      nd.insert(nd.begin(), outvecsize, 1);

      /* Calculate denominators in each dimension. */

      for(ptiter = wrtThese.begin(), ptiterend = wrtThese.end();
          ptiter != ptiterend; ptiter++)
        {
          /* Get outcome vector. */
          outvec = (outMap.find(*ptiter))->second;

          for(i = 0; i < outvecsize; i++)
        nd[i] += (*outvec)[i];
        }

      for(ptiter = forThese.begin(), ptiterend = forThese.end();
          ptiter != ptiterend; ptiter++)
        {
          outvec = (outMap.find(*ptiter))->second;
          sc = 0;

          for(i = 0; i < outvecsize; i++)
        sc += ((double) (*outvec)[i] / nd[i]);

          score.insert(typename map < ptype, double > :: value_type(*ptiter, sc));
        }
//      typename map < ptype , double > :: iterator sciter;
//      cout << "scm::sharingScore";
//      for(sciter = score.begin(); sciter != score.end(); sciter++)
//        cout << " " << sciter->first << "/" << sciter->second;
//      cout << endl;


    #ifdef MYDEBUG
      typename map < ptype *, double > :: iterator sciter;
      cout << "scm::sharingScore";
      for(sciter = score.begin(); sciter != score.end(); sciter++)
        cout << " " << sciter->first->id() << "/" << sciter->second;
      cout << endl;
    #endif
    }

    template <class ptype, class vtype > void EvoController::sharingScore(map < ptype , vector < vtype > * > *outMap,
                                    map < ptype , double > &score,
                                    set < ptype  > &forThese, /* Calculate score for these members. */
                                    set < ptype  > &wrtThese) /* Denominator w.r.t. these. */
    {
    	cout <<" ENTER##2" << endl;
      if(score.empty() == 0)
        die(__FILE__, __FUNCTION__, __LINE__, "score map should be empty");

      typename set < ptype > :: iterator ptiter, ptiterend;

      /* Denominator in each dimension. */
      vector < vtype > nd;

      vector < vtype > *outvec;
      double sc;
      int i;

      int outvecsize = (outMap->begin()->second)->size();

      /* Initialize to 1 so that do not divide by zero. */
      nd.insert(nd.begin(), outvecsize, 1);

      /* Calculate denominators in each dimension. */

      int qqq =0;
      for(ptiter = wrtThese.begin(), ptiterend = wrtThese.end();
          ptiter != ptiterend; ptiter++)
        {
    	  //printf(" %d \n", qqq++);
          /* Get outcome vector. */
          outvec = (outMap->find(*ptiter))->second;

          for(i = 0; i < outvecsize; i++)
        nd[i] += (*outvec)[i];
        }

//      printf("OutVecSize %d \n", outvecsize);
//      for(i = 0; i < outvecsize; i++)
//          printf("  %d",nd[i]);
//
//    printf("\n Scores: ");


      for(ptiter = forThese.begin(), ptiterend = forThese.end();
          ptiter != ptiterend; ptiter++)
        {

          outvec = (outMap->find(*ptiter))->second;
          sc = 0;

          for(i = 0; i < outvecsize; i++)
        sc += ((double) (*outvec)[i] / nd[i]);

    //      printf(" %f", sc);
          score.insert(typename map < ptype, double > :: value_type(*ptiter, sc));
        }
   //   printf("&\n");
 //     typename map < ptype , double > :: iterator sciter;
//      cout << "scm::sharingScore";
//      for(sciter = score.begin(); sciter != score.end(); sciter++)
//        cout << " " << sciter->first << "/" << sciter->second;
//      cout << endl;


    #ifdef MYDEBUG
      typename map < ptype *, double > :: iterator sciter;
      cout << "scm::sharingScore";
      for(sciter = score.begin(); sciter != score.end(); sciter++)
        cout << " " << sciter->first->id() << "/" << sciter->second;
      cout << endl;
    #endif
    }

void EvoController::selectPoints()
{


	//  DE->printTeamRewardMatrix();

	if (GPU_SELECT_POINTS)
	{
		thrust::host_vector<int> hToDel(DE->pointGenerationGapSize);
		thrust::fill(hToDel.begin(),hToDel.end(),-1);
		SelectPointsGPU(DE->pointPopulationSize,DE->teamPopulationSize,DE->teamRewardMatrix,DE->pointGenerationGapSize,&hToDel);

		for(int i =0; i < hToDel.size();i++)
		{
			pointsToDel.insert(hToDel[i]);
		}

		return;
	}

	     //  cout << "ENTER SEL POITNS " << endl;
	     //  cout << "ENTER SEL POITNS " << endl;

	     //  DE->printPointMatrix();

	        map <_pointIndex, double> score;

	        map<_pointIndex, vector<short> *> *dist = new map<_pointIndex, vector<short> *>() ;

	        (*dist)[-3] =  new vector < short>;
	    	delete (*dist)[-3];
	    	(*dist).erase(-3);
	        getDist(dist);

	        map < _pointIndex, vector < short > * > :: iterator diiter, diiterend;

	        set< _pointIndex> :: iterator poiter, poiterend;

	        set<_pointIndex> F,D;

	    	findParetoFront(dist,F,D,0.01);			// TODO


//	    	cout << "\nselPoints distinctions\n";
//	        for(diiter = dist->begin(); diiter != dist->end(); diiter++)
//	          {
//	            for(int _i=0; _i <diiter->second->size(); _i++ )
//	            {
//	                    if ((*diiter->second)[_i] == 0 ) { cout << "0";}
//	                    else {cout << (*diiter->second)[_i];}
//	                    if(_i % DE->teamPool->size() == DE->teamPool->size()-1 ){ cout << " ";}
//	            }
//	           cout << endl;
//	          }

	        cout << "selPoints F";
	        for(poiter = F.begin(); poiter != F.end(); poiter++)
	          cout << " " << (*poiter);
	        cout << endl;

	        cout << "selPoints D";
	        for(poiter = D.begin(); poiter != D.end(); poiter++)
	          cout << " " << (*poiter);
	        cout << endl;



	        int keep = DE->pointPool->size() - DE->pointGenerationGapSize;

	        _pointFrontSize = F.size();

	        int nrem;


	        if(F.size() == keep)
	          {
	            pointsToDel = D;
	          }
	        else if(F.size() < keep)
	          {
	            /* Must include some points from D. */

	    //        cout <<"1FSize: " << F.size() << endl;;
	    //        cout <<"1DSize: " << D.size() << endl;;
	    //        cout <<"1PS: " << DE->pointPool->size() << endl;;

	            sharingScore(dist, score, D, DE->pointPool->usedElements );
	    //        cout <<"2FSize: " << F.size() << endl;;
	    //        cout <<"2DSize: " << D.size() << endl;;
	            nrem = D.size() - (keep - F.size());
	    //        cout <<"3FSize: " << F.size() << endl;;
	    //        cout <<"3DSize: " << D.size() << endl;;
	            selPareto(D, score, nrem, pointsToDel);
	    //        cout <<"4FSize: " << F.size() << endl;;
	    //        cout <<"4DSize: " << D.size() << endl;;
	          }
	        else
	          {
	            /* Must discard some points from F (F.size() > keep). */

	            sharingScore(dist, score, F, F);
	          nrem = F.size() - keep;
	            selPareto(F, score, nrem, pointsToDel);

	            /* After selPareto toDel contains points from F to be
	           deleted, must add all points from D to toDel. */

	        pointsToDel.insert(D.begin(), D.end());
	          }

	        // MARK DELETED POINTS

	        /* Free allocated array. */

	        for(diiter = dist->begin(), diiterend = dist->end();
	            diiter != diiterend; diiter++)
	        {

	          delete diiter->second;
	        }




	    //    cout << " F: ";
	    //    for(set<_pointIndex>::iterator it= F.begin();it!= F.end(); it++)
	    //    {
	    //        cout << "    " << (*it);
	    //    }
	    //    cout << endl;
	    //    cout << " D: ";
	    //    for(set<_pointIndex>::iterator it= D.begin();it!= D.end(); it++)
	    //    {
	    //        cout << "    " << (*it);
	    //    }
	    //    cout << endl;

//	        cout << "TO DEL:::: ";
//	        for(set<_pointIndex>::iterator it= pointsToDel.begin();it!= pointsToDel.end(); it++)
//	        {
//	            cout << "    " << (*it);
//	        }
//	        cout << endl;




	    //    cout << "LEAVE SEL POITNS " << endl;

	    delete dist;
	    }
void EvoController::selectTeams()
{



	if(GPU_SELECT_TEAMS)
	{
		printf("Enter TSel");
		thrust::host_vector<int> hToDel(DE->teamGenerationGapSize);

		thrust::fill(hToDel.begin(),hToDel.end(),-1);
		hToDel.clear();
		SelectTeamsGPU(DE->teamPopulationSize,DE->pointPopulationSize, DE->teamRewardMatrix,DE->pointGenerationGapSize, &hToDel);
		printf(" $$>>: ");
		for(int i =0; i < hToDel.size();i++)
				{
					printf(" %d", hToDel[i]);
					teamsToDel.insert(hToDel[i]);
				}
		printf("Leave TSel");
		return;
	}
 //   cout << "Enter Select Teams " << endl;

    /* Team outcome maps. */
    map < _teamIndex , vector < _teamReward > * > outMap;
    map < _teamIndex, vector < _teamReward> *> :: iterator omiter, omiterend;

    /* Pareto fronts. */
    set < _teamIndex > F, D;
    set < _teamIndex > :: iterator fiter, diter;

    set < _teamIndex > :: iterator teiter, teiterend;
    set < _pointIndex> :: iterator poiter, poiterbegin, poiterend;

    /* Sharing score. */
    map < _teamIndex, double> score;


    int nrem;

    _teamReward outcome;

    /* Number of teams that make it to the next generation. */
    int keep = DE->teamPool->size() - DE->teamGenerationGapSize;

    /* Populate the outcome map. */

    teiter = DE->teamPool->begin();   teiterend = DE->teamPool->end();
    for(; teiter != teiterend; teiter++)
    {
        outMap[(*teiter)] = new vector<_teamReward>();

        int sum = 0;
     //   cout << "TM:"<< (*teiter) << "\t" ;
        poiter = DE->pointPool->begin();  poiterend = DE->pointPool->end();
        for(;poiter!= poiterend;poiter++)
        {
     //       cout  << DE->getReward((*poiter),(*teiter));
            sum +=  DE->getReward((*poiter),(*teiter));

            outMap[(*teiter)]->push_back(DE->getReward((*poiter),(*teiter)));
        }
    //   cout << "  = " << sum << endl;
    }

//    for(int j =0; j < DE->teamPopulationSize; j++)
//    {
//    	for(int i =0; i < DE->pointPopulationSize; i++)
//    	{
//    		cout << DE->teamRewardMatrix[i+j*DE->pointPopulationSize]  << ",";
//    	}
//    	cout << endl;
//    }
//    cout <<endl;


   // cout << "Find Team Front" << endl;
    findParetoFront(outMap,F,D,0.1);
    _teamFrontSize = F.size();

    cout << " TF: ";
    for(set<_teamIndex>::iterator it= F.begin();it!= F.end(); it++)
    {
        cout << "    " << (*it);
    }
    cout << endl;
    cout << " TD: ";
    for(set<_teamIndex>::iterator it= D.begin();it!= D.end(); it++)
    {
        cout << "    " << (*it);
    }
    cout << endl;

//        cout << "TeamsToDelSize= " << teamsToDel.size() << endl;
    if (F.size() == keep)
    {
        cout << " F ==" << keep <<  endl;
        teamsToDel = D;
    }
    else if(F.size() < keep)
    {
        /* Must include some points from D. */

   //    cout << " F <" << keep << endl;
        sharingScore(outMap, score, D, DE->teamPool->usedElements );

//        teiter = DE->teamPool->begin();   teiterend = DE->teamPool->end();
//        for(; teiter != teiterend; teiter++)
//        {
//            cout << " T"<<(*teiter)<<") " << score[(*teiter)] << endl;
//        }


        nrem = D.size() - (keep - F.size());
        selPareto(D, score, nrem, teamsToDel);
    }
    else
    {
   //     cout << " F >" << keep << endl;;
        /* Must discard some points from F (F.size() > keep). */

  //      cout << "TeamsToDelSize= " << teamsToDel.size() << endl;
        sharingScore(outMap, score, F, F);
//        cout << "::TeamSel Sharing::" << endl;
//        teiter = DE->teamPool->begin();   teiterend = DE->teamPool->end();
//        for(; teiter != teiterend; teiter++)
//        {
//            cout << " T"<<(*teiter)<<") " << score[(*teiter)] << endl;;
//        }

        nrem = F.size() - keep;
      //  cout << "nrem: " << nrem << endl;
        selPareto(F, score, nrem, teamsToDel);

//        cout << "::TeamSel Pareto::" << endl;
//        teiter = DE->teamPool->begin();   teiterend = DE->teamPool->end();
//        for(; teiter != teiterend; teiter++)
//        {
//            cout << " T"<<(*teiter)<<") " << score[(*teiter)] << endl;
//        }
        /* After selPareto toDel contains points from F to be
             deleted, must add all points from D to toDel. */

        cout << "TeamsToDelSize= " << teamsToDel.size() << endl;
        teamsToDel.insert(D.begin(), D.end());
    }

    for(teiter = teamsToDel.begin(), teiterend = teamsToDel.end();
        teiter != teiterend; teiter++)
    {
//        cout << "DeleteTeam:" << (*teiter) << endl;
        DE->teamPool->free(*teiter);
    }

    /* Free allocated array. */

    for(omiter = outMap.begin(), omiterend = outMap.end();
        omiter != omiterend; omiter++)
        delete omiter->second;



//    cout << "TO DEL";
//    for(set<_teamIndex>::iterator it= teamsToDel.begin();it!= teamsToDel.end(); it++)
//    {
//        cout << "    " << (*it);
//    }
//    cout << endl;
}



void EvoController::cleanup()
{

   // cout << "======================" << "  Clean Up  " << "======================" << endl;
   // cout << "Deleting " << pointsToDel.size()  << " points" << endl;


    set < _pointIndex > :: iterator poiter, poiterend;

    for (poiter = pointsToDel.begin(), poiterend = pointsToDel.end(); poiter != poiterend; poiter++)
    {
         _pointIndex pindex = (*poiter);
         for(int i=0;i<DE->pointDim; i++)
         {
             DE->pointMatrix[pindex*DE->pointDim + i] = -2;
         }
         DE->pointPool->free((*poiter));
         if (DE->pointAge[*poiter] != time) { _pdom++;}

         DE->usedPointLabelCounts[DE->pointLabelMatrix[(*poiter)]]--;

    }
    pointsToDel.clear();


    set < _teamIndex > :: iterator iter, iterend;

    for (iter = teamsToDel.begin(), iterend = teamsToDel.end(); iter != iterend; iter++)
    {
         _teamIndex tindex = (*iter);
       //  cout << "TEAMINDEX: " << tindex << endl;
       int lid;
         for(int i=0;i<DE->maxTeamSize; i++)
         {

             lid = DE->teamMatrix[tindex*DE->maxTeamSize +i];
             if( --(DE->learnerReferenceCounts[lid]) == 0)
             {
   //          cout << "   >>>T(" << tindex <<") Delete LID: " << lid << endl;
                 _learner* L = DE->getPtrToLearnerMatrix(lid);
                 for(int i=1;i< L[0]+1; i++)
                 {
                     L[i] =0;
                 }
                 L[0] = 0;
                 DE->learnerPool->free(lid);
             }
             DE->teamMatrix[tindex*DE->maxTeamSize + i] = -2;
         }

       //  cout << "MDOM: " << _mdom << "   <--  Age:" << DE->teamAge[*iter] << " T: " << time << " " << (DE->teamAge[*iter] != time) << endl;
     //    if (DE->teamAge[*iter] != time) { _mdom++;}
         DE->teamPool->free((*iter));
    }
    teamsToDel.clear();


/*TODO CLEANUP LEARNERS FULL MARK N SWEEP*/



}


void EvoController::testTeams_gpu()
{

    _point* pointSet = DE->DS->trainingPointSet;
    int     pointSetSize = DE->DS->size_training;

    cout << "======================" << " Testing Teams  "  << pointSetSize << "======================" << endl;
    set<_teamIndex>::iterator it;

    double bestScore = -1;
    int    bestTeam = -1;

    _learner* usedLearners = new _learner[DE->learnerLength*DE->maxTeamSize+10];
    _learnerBid* teamBids = new _learnerBid[ (DE->learnerPool->maxUsedIndex()+1) * pointSetSize+100];



    	TestLearners(DE->learnerPool->maxUsedIndex()+1,DE->learnerLength,  DE->learnerMatrix, teamBids,  DE->DS->size_training, DE->DS->trainingPointSet, DE->pointDim);

    	//for(int i=0; i < DE->learnerPool->maxUsedIndex()+1; i++)
//    	for(int i=0; i < 10; i++)
//    	{
//    		for(int j=0; j < pointSetSize; j++)
//    		{
//    			cout << " " << teamBids[i*pointSetSize+j];
//    		}
//    		cout << endl;
//    	}

    for(it=DE->teamPool->begin();it != DE->teamPool->end(); it++)
    {

        _team* T = DE->getPtrToTeamMatrix(*it);

        vector < long > cm;
        long matDim = DE->DS->numLabels() * DE->DS->numLabels();
        cm.insert(cm.begin(), matDim, 0);

        double score;

        int hit =0;
        int mis =0;


       // cout << "teamSize: " << teamSize <<  "  PSize:      " << pointSetSize << endl;
        //pointSetSize = 1280;
   //TODO: Selective Eval? or Batch
    //       TestLearners(teamSize,usedLearners,teamBids,pointSetSize,pointSet);
       // teamBids = &DE->learnerBidMatrix[(*it)* DE->pointPopulationSize];

       // if (*it ==0){
       //         for(int pid=0; pid < pointSetSize; pid++)
       //         {
       //                 teamBids[5*pointSetSize+pid] = pid;
       //         }
       // }
       /* cout << "SAmple " << endl   ;
        for(int ts =0; ts < teamSize; ts++)
        {
                for(int ps=0; ps < 8; ps++)
                {

                    cout << "   " << teamBids[ts*pointSetSize+ps];
                }
                cout << endl;
        }
*/

        int lid;

            for(int pid = 0; pid < pointSetSize; pid++)
        {

            _learnerBid maxBid =-1;
            _learnerId maxBidder =-1;
            for(int i =0; T[i] != -1 && i < DE->maxTeamSize; i++)
            {
                _learnerBid* Bid = &teamBids[T[i]*pointSetSize];
//                cout << "          " << i << " " << T[i] << " " << pid;
//                cout << "          " << T[i];
//                cout << flush;
//                cout << " ---> " << Bid[pid] << endl;;
                if (Bid[pid] > maxBid)
                {
                    maxBid = Bid[pid];
                    maxBidder = T[i];
                }


            }
           //:     FIND MAX BIDS CMPARE
           //cout << "T: " << *it << "  P: " << pid << "  ";
           // cout << "MAXBID " << maxBid << "  (" << T[maxBidder] << ") ";
            _pointLabel pointLabel = DE->DS->trainingPointSetLabels[pid];
            _learnerAction action =  DE->getLearnerAction(maxBidder);

           // cout << " Label: " << pointLabel << " Action " << action << endl;
            cm[pointLabel * DE->DS->numLabels() + action]++;


            if( pointLabel == action){
                hit++;
            }else{
                mis++;
            }

        }
        score = 0;
        cout << *it << ") :::# ";

        for(int i = 0; i < DE->DS->numLabels(); i++){
            score += (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTrainingLabelCount(i);
            cout <<  cm[i * DE->DS->numLabels() + i] << "/" <<DE->DS->getTrainingLabelCount(i) << "    ";
        }
        score = (double) score/ DE->DS->numLabels();
        long partfp;
        cout << "explicitEnv::test Training ";
        cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
        cout << " score " << score << " cm";

        int i,j;
        for(i = 0; i <= DE->DS->numLabels(); i++)
            cout << " " << i;
        for(i = 0; i < matDim; i++)
            cout << " " << cm[i];
        cout << " dr fr";
        for(i = 0; i < DE->DS->numLabels(); i++)
        {
            cout << " " << (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTrainingLabelCount(i);

            partfp = 0;
            for(j = 0; j < DE->DS->numLabels(); j++)
                if(i != j)
                    partfp += cm[j * DE->DS->numLabels() + i];

            cout << " " << (double) partfp / (DE->DS->getTrainingSetSize() - DE->DS->getTrainingLabelCount(i));
        }
        cout << " " << endl;

        if ( score > bestScore) {

            bestScore = score;
            bestTeam = (*it);
        }
    }

    cout << " =============== " << endl;
        delete(usedLearners);
        delete(teamBids);
    cout << "BESTTeam: " << bestTeam <<endl;
      testTeamAgainstTestingSet_gpu(bestTeam);



}

double EvoController::testTeamAgainstTestingSet_gpu(_teamIndex tindex)
{

    _point* pointSet = DE->DS->testingPointSet;
    int     pointSetSize = DE->DS->size_testing;

    set<_teamIndex>::iterator it;
    _learnerBid* teamBids = (_learnerBid*)malloc(sizeof(_learnerBid)*DE->maxTeamSize * pointSetSize);

    _learner* usedLearners = new _learner[DE->learnerLength*DE->maxTeamSize];

     int teamSize=0;
     _team* T = DE->getPtrToTeamMatrix(tindex);

     for(int tid=0; T[tid] !=-1 && tid < DE->maxTeamSize; tid++)
     {
            teamSize++;
            _learner* L = DE->getPtrToLearnerMatrix(T[tid]);
            for(int i=0; i<DE->learnerLength;i++)
            {
                usedLearners[tid*DE->learnerLength+i] = L[i];
            }
        }


        vector < long > cm;
        long matDim = DE->DS->numLabels() * DE->DS->numLabels();
        cm.insert(cm.begin(), matDim, 0);

        double score;

        int hit =0;
        int mis =0;


       // cout << "teamSize: " << teamSize <<  "  PSize:      " << pointSetSize << endl;
        //pointSetSize = 1280;
        //EvaluateLearners(teamSize,usedLearners,teamBids,pointSetSize,pointSet);
        TestLearners(teamSize,DE->learnerLength,usedLearners, teamBids, pointSetSize, pointSet, DE->pointDim);


        int lid;

            for(int pid = 0; pid < pointSetSize; pid++)
        {

            _learnerBid maxBid =-1;
            _learnerId maxBidder =-1;
            for(int lid =0; lid < teamSize; lid++)
            {
                _learnerBid* Bid = &teamBids[lid*pointSetSize];
              //  cout << "          " << lid << " ---> " << Bid[pid] << endl;;
                if (Bid[pid] > maxBid)
                {
                    maxBid = Bid[pid];
                    maxBidder = lid;
                }


            }
           //:     FIND MAX BIDS CMPARE
         //   cout << "MAXBID " << maxBid << "  (" << T[maxBidder] << ") ";
            _pointLabel pointLabel = DE->DS->testingPointSetLabels[pid];
            _learnerAction action =  DE->getLearnerAction(T[maxBidder]);

         //i   cout << " Label: " << pointLabel << " Action " << action << endl;
            cm[pointLabel * DE->DS->numLabels() + action]++;


            if( pointLabel == action){
                hit++;
            }else{
                mis++;
            }

        }
        score = 0;
        cout << tindex << ") :::# ";

        for(int i = 0; i < DE->DS->numLabels(); i++){
            score += (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);
            cout <<  cm[i * DE->DS->numLabels() + i] << "/" <<DE->DS->getTestingLabelCount(i) << "    ";
        }
        score = (double) score/ DE->DS->numLabels();
        long partfp;
        cout << "explicitEnv::test Training ";
        cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
        cout << " score " << score << " cm";

        int i,j;
        for(i = 0; i <= DE->DS->numLabels(); i++)
            cout << " " << i;
        for(i = 0; i < matDim; i++)
            cout << " " << cm[i];
        cout << " dr fr";
        for(i = 0; i < DE->DS->numLabels(); i++)
        {
            cout << " " << (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);

            partfp = 0;
            for(j = 0; j < DE->DS->numLabels(); j++)
                if(i != j)
                    partfp += cm[j * DE->DS->numLabels() + i];

            cout << " " << (double) partfp / (DE->DS->getTestingSetSize() - DE->DS->getTestingLabelCount(i));
        }
        cout << " " << endl;
        return score;
}



void EvoController::testTeams_cpu()
{
    cout << "======================" << " Testing Teams  CPU" << "======================" << endl;
    set<_teamIndex>::iterator it;

 //   DE->printTeamMatrix();
 //   DE->printLearnerMatrix();
    double bestScore = -1;
    int    bestTeam = -1;

    for(it=DE->teamPool->begin();it != DE->teamPool->end(); it++)
    {
        double score = testTeamAgainstTrainingSet(*it);
        if ( score > bestScore) {

            bestScore = score;
            bestTeam = (*it);
        }
    }
    cout << " =============== " << endl;
//    for(it=DE->teamPool->begin();it != DE->teamPool->end(); it++)
//    {
//        testTeamAgainstTestingSet(*it);
//    }
    cout << "BESTTeam: " << bestTeam <<endl;
testTeamAgainstTestingSet_cpu(bestTeam);}


double EvoController::testTeamAgainstTestingSet_cpu(_teamIndex tindex)
{

    _team* T = DE->getPtrToTeamMatrix(tindex);
    _point P[DE->pointDim];

    for(int i =0; i < DE->pointDim;i++)
    { P[i] =0;}

    _learner* L;



     vector < long > cm;
     long matDim = DE->DS->numLabels() * DE->DS->numLabels();
     cm.insert(cm.begin(), matDim, 0);

     double score;

    int hit =0;
    int mis =0;

    for(int pid=0; pid < DE->DS->getTestingSetSize();pid++)
    {
        DE->DS->copyTestingPointToMatrix(pid,P);

        int arrayIndex=0;
        int action;

        _learnerBid maxBid = -1;
        _learnerIndex maxBidder;

////       for(int j=0; j < DE->pointDim;j++)
////        {
////           cout << "  " << P[j];
////        }
////        cout << " :: " << DE->DS_test->pointSetLabels[pid] << endl;

        //cout << "ArrayIndex:" << arrayIndex << " " << DE->learnerLength << endl;
        for(_learnerIndex lindex=T[arrayIndex]; lindex != -1 && arrayIndex < DE->maxTeamSize ; lindex = T[++arrayIndex])
        {
          //  cout << "LINDEX2:" << lindex << endl;
            L = DE->getPtrToLearnerMatrix(lindex);

            _learnerBid bid;
            cLearnerEvalSingle(L, bid, P, DE->learnerLength, DE->pointDim );
            // cout <<" L:" << lindex << "P: " << pid << " Bid: " << bid <<endl;


            if (bid > maxBid)
            {
                maxBid = bid;
                maxBidder = lindex;
            }


        }

        _pointLabel pointLabel = DE->DS->testingPointSetLabels[pid];
        action =  DE->getLearnerAction(maxBidder);

        cm[pointLabel * DE->DS->numLabels() + action]++;


        if( pointLabel == action){
            hit++;
        }else{
            mis++;
        }


    }

    score = 0;
    cout << tindex << ") :::# ";

    for(int i = 0; i < DE->DS->numLabels(); i++){
      score += (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);
      cout <<  cm[i * DE->DS->numLabels() + i] << "/" <<DE->DS->getTestingLabelCount(i) << "    ";
}
    score = (double) score/ DE->DS->numLabels();



//    cout << "explicitTest " << tindex ;
//    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
//    cout << endl;


long partfp;
    cout << "explicitEnv::test Training2 ";
    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
    cout << " score " << score << " cm";

    int i,j;
    for(i = 0; i <= DE->DS->numLabels(); i++)
      cout << "." << i;
    for(i = 0; i < matDim; i++)
      cout << " " << cm[i];
    cout << " dr fr";
    for(i = 0; i < DE->DS->numLabels(); i++)
      {
        cout << " " << (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);

        partfp = 0;
        for(j = 0; j < DE->DS->numLabels(); j++)
      if(i != j)
        partfp += cm[j * DE->DS->numLabels() + i];

        cout << " " << (double) partfp / (DE->DS->getTestingSetSize() - DE->DS->getTestingLabelCount(i));
      }
    cout << " " << endl;
    return score;

}



double EvoController::testTeamAgainstTrainingSet(_teamIndex tindex)
{

 //   cout << " EValTrain " << tindex << endl;
    _team* T = DE->getPtrToTeamMatrix(tindex);
   //  cout << "T:"<<T<<endl;
    _point P[DE->pointDim];
    for(int i =0; i < DE->pointDim;i++)
    { P[i] =0;}
    _learner* L;



     vector < long > cm;
     long matDim = DE->DS->numLabels() * DE->DS->numLabels();
     cm.insert(cm.begin(), matDim, 0);

     double score;

    int hit =0;
    int mis =0;


   //  cout << "Ti:"<<T<<endl;

    for(int pid=0; pid < DE->DS->getTrainingSetSize();pid++)
    {

      //  cout << "Ts:"<<T<<endl;
      //  cout << "PID: " << pid << endl;
        DE->DS->copyTrainingPointToMatrix(pid,P);
     //   cout << "Tq:"<<T<<endl;

        int arrayIndex=0;
        int action;

        _learnerBid maxBid = -1;
        _learnerIndex maxBidder;

////       for(int j=0; j < DE->pointDim;j++)
////        {
////           cout << "  " << P[j];
////        }
////        cout << " :: " << DE->DS_test->pointSetLabels[pid] << endl;

//
//        cout << "PID: " << pid << "  " << arrayIndex<< endl;
//     cout << "Ta:"<<T<<endl;
//        cout <<"QWE" << T[arrayIndex];
        for(_learnerIndex lindex=T[arrayIndex]; lindex != -1 && arrayIndex < DE->maxTeamSize; lindex = T[++arrayIndex])
        {
//            cout << "arrayIndex:" << arrayIndex << " of " << DE->maxTeamSize << " Linedex:" << lindex <<  endl;
            L = DE->getPtrToLearnerMatrix(lindex);

            _learnerBid bid;
            cLearnerEvalSingle(L, bid, P, DE->learnerLength, DE->pointDim );
            // cout <<" L:" << lindex << "P: " << pid << " Bid: " << bid <<endl;


           //     cout << "          " << lindex << " ---> " << bid << endl;;
            if (bid > maxBid)
            {
                maxBid = bid;
                maxBidder = lindex;
            }


//        cout << "Tb"<<lindex << ": "<<T<<endl;
        }

        _pointLabel pointLabel = DE->DS->trainingPointSetLabels[pid];
        action =  DE->getLearnerAction(maxBidder);

        cm[pointLabel * DE->DS->numLabels() + action]++;
     //   cout << "T: " << tindex << "  P: " << pid << "  ";
     //   cout << "MAXBID " << maxBid << "  (" << maxBidder << ") ";

      //  cout << " Label: " << pointLabel << " Action " << action << endl;


        if( pointLabel == action){
            hit++;
        }else{
            mis++;
        }


//     cout << "Te:"<<T<<endl;
    }

    score = 0;
    cout << tindex << ") :::# ";

    for(int i = 0; i < DE->DS->numLabels(); i++){
      score += (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTrainingLabelCount(i);
      cout <<  cm[i * DE->DS->numLabels() + i] << "/" <<DE->DS->getTrainingLabelCount(i) << "    ";
}
    score = (double) score/ DE->DS->numLabels();



//    cout << "explicitTest " << tindex ;
//    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
//    cout << endl;


long partfp;
    cout << "explicitEnv::test Training ";
    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
    cout << " score " << score << " cm";

    int i,j;
    for(i = 0; i <= DE->DS->numLabels(); i++)
      cout << " " << i;
    for(i = 0; i < matDim; i++)
      cout << " " << cm[i];
    cout << " dr fr";
    for(i = 0; i < DE->DS->numLabels(); i++)
      {
        cout << " " << (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTrainingLabelCount(i);

        partfp = 0;
        for(j = 0; j < DE->DS->numLabels(); j++)
      if(i != j)
        partfp += cm[j * DE->DS->numLabels() + i];

        cout << " " << (double) partfp / (DE->DS->getTrainingSetSize() - DE->DS->getTrainingLabelCount(i));
      }
    cout << " " << endl;
    return score;

}


void EvoController::testTeam(_teamIndex tindex)
{

    _team* T = DE->getPtrToTeamMatrix(tindex);
    _point P[DE->pointDim];
    _learner* L;



     vector < long > cm;
     long matDim = DE->DS->numLabels() * DE->DS->numLabels();
     cm.insert(cm.begin(), matDim, 0);

     double score;

    int hit =0;
    int mis =0;

    for(int pid=0; pid < DE->DS->getTestingSetSize();pid++)
    {
        DE->DS->copyTestingPointToMatrix(pid,P);

        int arrayIndex=0;
        int action;

        _learnerBid maxBid = -1;
        _learnerIndex maxBidder;

////       for(int j=0; j < DE->pointDim;j++)
////        {
////           cout << "  " << P[j];
////        }
////        cout << " :: " << DE->DS_test->pointSetLabels[pid] << endl;

        for(_learnerIndex lindex=T[arrayIndex]; lindex != -1; lindex = T[++arrayIndex])
        {
            L = DE->getPtrToLearnerMatrix(lindex);


           // cout << " Lindex" << lindex << endl;
            _learnerBid bid;
            cLearnerEvalSingle(L, bid, P, DE->learnerLength, DE->pointDim );
            // cout <<" L:" << lindex << "P: " << pid << " Bid: " << bid <<endl;


            if (bid > maxBid)
            {
                maxBid = bid;
                maxBidder = lindex;
            }


        }

        _pointLabel pointLabel = DE->DS->testingPointSetLabels[pid];
        action =  DE->getLearnerAction(maxBidder);

        cm[pointLabel * DE->DS->numLabels() + action]++;


        if( pointLabel == action){
            hit++;
        }else{
            mis++;
        }


    }

    score = 0;
    cout << tindex << ") :::# ";
    for(int i = 0; i < DE->DS->numLabels(); i++){
      score += (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);
      cout <<  cm[i * DE->DS->numLabels() + i] << "/" <<DE->DS->getTestingLabelCount(i) << "    ";
    }
    score = (double) score/ DE->DS->numLabels();



//    cout << "explicitTest " << tindex ;
//    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
//    cout << endl;


long partfp;
    cout << "explicitEnv::test Testing";
    cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
    cout << " score " << score << " cm";

    int i,j;
    for(i = 0; i <= DE->DS->numLabels(); i++)
      cout << "." << i;
    for(i = 0; i < matDim; i++)
      cout << " " << cm[i];
    cout << " dr fr";
    for(i = 0; i < DE->DS->numLabels(); i++)
      {
        cout << " " << (double) cm[i * DE->DS->numLabels() + i] / DE->DS->getTestingLabelCount(i);

        partfp = 0;
        for(j = 0; j < DE->DS->numLabels(); j++)
      if(i != j)
        partfp += cm[j * DE->DS->numLabels() + i];

        cout << " " << (double) partfp / (DE->DS->getTestingSetSize() - DE->DS->getTestingLabelCount(i));
      }
    cout << " " << endl;

}
//          action = tm->getAction(state,-1);

//          cm[labelIndex * numLabels() + action]++;

//          if(labelIndex == action)
//        {
//          /* Correct. */
//          hit++;
//        }
//          else
//        {
//          /* Inorrect. */
//          mis++;
//        }
//        }

//      /* Calculate score. */

//      score = 0;

//      for(i = 0; i < numLabels(); i++)
//        score += (double) cm[i * numLabels() + i] / getLabelCount(getLabel(i));

//      score = (double) score / numLabels();

//      /* Print out results. */

//      cout << "explicitEnv::test " << prefix << " " << _setName;
//      cout << " hit " << hit << " mis " << mis << " rate " << (double) hit / (hit + mis);
//      cout << " score " << score << " cm";
//      for(i = 0; i < numLabels(); i++)
//        cout << " " << getLabel(i);
//      for(i = 0; i < matDim; i++)
//        cout << " " << cm[i];
//      cout << " dr fr";
//      for(i = 0; i < numLabels(); i++)
//        {
//          cout << " " << (double) cm[i * numLabels() + i] / getLabelCount(getLabel(i));

//          partfp = 0;
//          for(j = 0; j < numLabels(); j++)
//        if(i != j)
//          partfp += cm[j * numLabels() + i];

//          cout << " " << (double) partfp / (size() - getLabelCount(getLabel(i)));
//        }
//      cout << " " << *tm << endl;

//      return score;



void EvoController::testTeams()
{
	useGPU = false;
    if (useGPU)
    {
        testTeams_gpu();
    }else{
        testTeams_cpu();
    }
}

void EvoController::stats(int t)
{
    int sumTeamSizes = 0;
    int nrefs = 0;
    int sumNumOutcomes = 0;

    /* Labels in _P. */
    vector < long > labelCount;
    /* Actions in _L. */
    vector < long > actionCount;

    /* Quartiles for team sizes. */
    vector < int > msize;
    int mq1 = (int) (DE->teamPool->size() * 0.25);
    int mq2 = (int) (DE->teamPool->size() * 0.5);
    int mq3 = (int) (DE->teamPool->size() * 0.75);

    set < _pointIndex> :: iterator poiter;
    set < _learnerIndex> :: iterator leiter;
    set < _teamIndex > :: iterator teiter;

    labelCount.insert(labelCount.begin(), DE->DS->labelVector.size(), 0);
    actionCount.insert(actionCount.begin(), DE->DS->labelVector.size(), 0);

    for(poiter = DE->pointPool->begin(); poiter != DE->pointPool->end(); poiter++){
      labelCount[DE->pointLabelMatrix[(*poiter)]]++;
    }
    for(leiter = DE->learnerPool->begin(); leiter != DE->learnerPool->end(); leiter++)
      {
        actionCount[DE->getLearnerAction((*leiter))]++;
        nrefs += DE->learnerReferenceCounts[(*leiter)];
      }

    for(teiter = DE->teamPool->begin(); teiter != DE->teamPool->end(); teiter++)
      {
          _team* T = DE->getPtrToTeamMatrix(*teiter);
          int _i = 0;
          while(T[_i++] >0){}
        msize.push_back(_i);
        sumTeamSizes += _i;
        sumNumOutcomes += 0;
      }

    sort(msize.begin(), msize.end());

    cout << "scm::stats " << DE->_seed << " " << t;
    cout << " mdom " << _mdom;
    cout << " pdom " << _pdom;
    cout << " Lsize " << DE->learnerPool->size();

    cout << " labels";
    for(int i = 0; i < DE->DS->numLabels(); i++)
      cout << " " << i;

    cout << " points";
    for(int i = 0; i < labelCount.size(); i++)
      cout << " " << labelCount[i];

    cout << " learners";
    for(int i = 0; i < actionCount.size(); i++)
      cout << " " << actionCount[i];

    cout << " msize " << msize[mq1] << " " << msize[mq2] << " " << msize[mq3];

    cout << " pointFront " << _pointFrontSize << " teamFront " << _teamFrontSize;

    cout << endl;

    /* Preform the following check to make sure that the number of
       references recoreded in the learners is actually equal to the sum
       of the team sizes and that the outcome counts equal the number of
       points by the number of teams. */

    if(sumTeamSizes != nrefs || sumNumOutcomes != (DE->teamPool->size() - DE->teamGenerationGapSize) * (DE->pointPool->size()- DE->pointGenerationGapSize))
        cout << "something does not add up" << endl;

        //die(__FILE__, __FUNCTION__, __LINE__, "something does not add up");


}
