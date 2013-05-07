#include "DataEngine.hpp"
//#include "GpuController_simple.cuh"
#include <cstring>
#include <cmath>
#include <cassert>
//#define MYDEBUG
#define EMPTY_INSTRUCTION 0

DataEngine::DataEngine(map < string,string> &args)
{
   LoadArgs(args);
   initDataSets(args);

   learnerPopulationSize =  teamPopulationSize * maxTeamSize-1;    // NEEDS WORK
   actionCount = DS->labelVector.size();
   //pointDim = setDim;
   learnerLength = maxProgSize +1;

   InitializeMemory();


}
void DataEngine::LoadTestRun(string pfile, string lfile, string tfile)
{

if ( pfile != "")
{
   printPointMatrix();

   memset(pointMatrix,0,bytesize_PointMatrix);
   usedPointIds.clear();

   for (int j=0; j < DS->labelVector.size();j++)
   {
        usedPointLabelCounts[DS->labelVector[j]] = 0;

   }
   ifstream pointfile ;
   pointfile.open (pfile.c_str(), ios::in);

   pointPool = new FreeMap<int>(0,pointPopulationSize);

   if(pointfile.good()){


       int pid;
       for(int i=0; i < pointPopulationSize;i++)
       {
           pointfile >> pid;
       _pointId matrixIndex =  pointPool->getNextFree();
           cout <<" Pid: " << pid << " " << matrixIndex <<  endl;

           _point* pm = &pointMatrix[matrixIndex*pointDim];
           DS->copyTrainingPointToMatrix(pid, pm);

           pointIndexMap[pid] = matrixIndex;
           indexPointMap[matrixIndex] = pid;
           pointAge[pid] = 0;
            // Copy Label
           _pointLabel label =  DS->trainingPointSetLabels[pid];
           cout << "POI::: " << pid <<"    - -   " <<  label <<endl;
           pointLabelMatrix[matrixIndex] = label;
           usedPointLabelCounts[label]++;
       }

   }
   printPointMatrix();
}

if(lfile != "" and tfile != "")
{
learnerReferenceCounts.clear();
   printPointLabelMatrix();








    printLearnerMatrix();

    memset(learnerMatrix,0,bytesize_LearnerMatrix);
    ifstream learnfile ;
    learnfile.open (lfile.c_str(), ios::in);


    learnerPool = new FreeMap<int>(0,learnerPopulationSize);

    map<int,int> II;

    while(learnfile.good()){
        string x;
        int id,action;
        int inst_count = 0;

        if(learnerPool->hasFreeIndex() == false)
        {
                cout << " BORKETD" <<endl;

        }
        int lIndex = learnerPool->getNextFree();
        cout << "LINDEX:" <<  lIndex << "   " ;
        learnerReferenceCounts[lIndex] = 0;
        _learner* L = &learnerMatrix[lIndex*learnerLength];

        cout << " 1 " ;

        learnfile >> id;
        cout << " 2("<<id<<") " ;
        learnfile >> action;
        cout << " 3 " ;
        int i = 1;
        L[0] = i;
        cout << " 4 " ;
        for(; i < learnerLength+1;i++)
        {
        cout << " 5("<<i<<") " ;
            learnfile >> x;
        cout << " 6 " ;
            if (x == "#")
            {
                cout << " ###7 " ;
                break;
            }

        cout << " 8 " ;

            L[i] = atoi(x.c_str());
        cout << " 9 " ;

        }

        cout << " A " ;

        if(learnfile.good()){
        cout << "\nMApping" << id <<"(" << id <<") " << lIndex << endl;
        II[id] = lIndex;
            L[0] = i-1;
            learnerActionMatrix[lIndex] = action;
			cout << "Learner:: " << lIndex << " ACTION:" << action <<endl;
        }else{
            L[0] =0;
        }
    }
    printLearnerMatrix();
    printTeamMatrix();

    memset(teamMatrix,-1,bytesize_TeamMatrix);
    ifstream teamfile ;
    teamfile.open (tfile.c_str(), ios::in);

	cout << "@@@@@@@@@@@@@@@@@@@   " << teamPopulationSize << endl;
    teamPool = new FreeMap<int>(0,teamPopulationSize);


    int team_index=0;
    while(teamfile.good()){
        string x,id;
        int member_count = 0;
		cout << " #### get NExt \n";
        _team* T = &teamMatrix[teamPool->getNextFree()*maxTeamSize];


        teamfile >> id;

        int i = 0;
            	teamfile >> x;
            	cout << " " << x << "|" <<  II[atoi(x.c_str())] << "  ";
        bool fill =0;
        for(; i < maxTeamSize;i++)
        {
            if (x == "#")
                fill= true;

            if (fill == false){
                learnerReferenceCounts[   II[atoi(x.c_str())] ] ++;
                T[i] = II[atoi(x.c_str())];
				cout << "^" ;
            	teamfile >> x;
            	cout << " " << x << "|" <<  II[atoi(x.c_str())] << "  ";
            }
            else{ T[i] = -1;}

        }
        cout << "^^^\n";

        if(teamfile.good()){
            team_index++;

        }else{

        }
    }
    printTeamMatrix();
}
}

bool DataEngine::LoadArgs(map < string,string> &args)
    {
      map < string, string > :: iterator maiter;

      if((maiter = args.find("Psize")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg Psize");

      pointPopulationSize = stringToInt(maiter->second);

      if((maiter = args.find("Msize")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg Msize");

      teamPopulationSize = stringToInt(maiter->second);

      if((maiter = args.find("pd")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pd");

      PROB_LearnerDeletion = stringToFloat(maiter->second);

      if((maiter = args.find("pa")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pa");

      PROB_LearnerAddition = stringToFloat(maiter->second);

      if((maiter = args.find("mua")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg mua");

       PROB_ActionMutation = stringToFloat(maiter->second);

      if((maiter = args.find("omega")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg omega");

      maxTeamSize = stringToInt(maiter->second);

      if((maiter = args.find("t")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg t");

      trainingGenerations = stringToLong(maiter->second);

      if((maiter = args.find("Pgap")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg Pgap");

      pointGenerationGapSize = stringToInt(maiter->second);

      if((maiter = args.find("Mgap")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg Mgap");

      teamGenerationGapSize = stringToInt(maiter->second);

      if((maiter = args.find("maxProgSize")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg maxProgSize");

      maxProgSize = stringToInt(maiter->second);

      if((maiter = args.find("pBidDelete")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pBidDelete");

      PROB_BidDeletion = stringToFloat(maiter->second);

      if((maiter = args.find("pBidAdd")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pBidAdd");

      PROB_BidAddition = stringToFloat(maiter->second);

      if((maiter = args.find("pBidSwap")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pBidSwap");

      PROB_BidSwap = stringToFloat(maiter->second);

      if((maiter = args.find("pBidMutate")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg pBidMutate");

      PROB_BidMutate = stringToFloat(maiter->second);

      if((maiter = args.find("statMod")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg statMod");

      statMod = stringToLong(maiter->second);

      if((maiter = args.find("seed")) == args.end())
        die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg seed");

      _seed = stringToLong(maiter->second);

      cout << "arg Psize " << pointPopulationSize << endl;
      cout << "arg Msize " << teamPopulationSize << endl;
      cout << "arg pd " << PROB_LearnerDeletion << endl;
      cout << "arg pa " << PROB_LearnerAddition << endl;
      cout << "arg mua " << PROB_ActionMutation << endl;
      cout << "arg omega " << maxTeamSize << endl;
      cout << "arg t " << trainingGenerations << endl;
      cout << "arg Pgap " << pointGenerationGapSize << endl;
      cout << "arg Mgap " << teamGenerationGapSize << endl;

      cout << "arg maxProgSize " << maxProgSize << endl;
      cout << "arg pBidDelete " << PROB_BidDeletion << endl;
      cout << "arg pBidAdd " << PROB_BidAddition << endl;
      cout << "arg pBidSwap " << PROB_BidSwap << endl;
      cout << "arg pBidMutate " << PROB_BidMutate << endl;

      cout << "arg statMod " << statMod << endl;
      cout << "arg seed " << _seed << endl;

      if(pointPopulationSize < 2)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg Psize < 2");

      if(teamPopulationSize < 2)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg Msize < 2");

      if(PROB_LearnerDeletion < 0 || PROB_LearnerDeletion > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pd < 0 || pd > 1");

      if(PROB_LearnerAddition < 0 || PROB_LearnerAddition > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pa < 0 || pa > 1");

      if(PROB_ActionMutation < 0 || PROB_ActionMutation > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg mua < 0 || mua > 1");

      if(maxTeamSize < 2)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg omega < 2");

      if(trainingGenerations< 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg t < 1");

      if(pointGenerationGapSize < 1 || pointGenerationGapSize >= pointPopulationSize)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg Pgap < 1 || Pgap >= Psize");

      if( teamGenerationGapSize < 1 || teamGenerationGapSize >= teamPopulationSize || teamGenerationGapSize % 2 != 0)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg Mgap < 1 || Mgap >= Msize || Mgap % 2 != 0");

      if(maxProgSize < 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg _maxProgSize < 1");

      if(PROB_BidDeletion < 0 || PROB_BidDeletion > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pBidDelete < 0 || pBidDelete > 1");

      if(PROB_BidAddition < 0 || PROB_BidAddition > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pBidAdd < 0 || pBidAdd > 1");

      if(PROB_BidSwap < 0 || PROB_BidSwap > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pBidSwap < 0 || pBidSwap > 1");

      if(PROB_BidMutate < 0 || PROB_BidMutate > 1)
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg pBidMutate < 0 || pBidMutate > 1");



      return 1;
}

void DataEngine::InitializeMemory()
{

/*	cout << " MAXTEAMSIZE: " << maxTeamSize << endl;
	cout << " TEAMPOP: " << teamPopulationSize << endl;
	cout << " LEArner	POP: " << learnerPopulationSize << endl;
	cout << " POINtPOP: " << pointPopulationSize << endl;
	cout << " LearnLEgnth: " << learnerLength << endl;
	cout << " Point Dim: " << pointDim<< endl;
*/
    bytesize_TeamMatrix                 =  sizeof(int) * maxTeamSize * teamPopulationSize ;
    bytesize_TeamLengthMatrix           =  sizeof(int) * teamPopulationSize ;
    bytesize_TeamRewardMatrix           =  sizeof(_teamReward) * teamPopulationSize * pointPopulationSize;
    bytesize_LearnerMatrix              =  sizeof(short) * learnerLength * learnerPopulationSize ;
    bytesize_LearnerBidMatrix              =  sizeof(_learnerBid) * pointPopulationSize * learnerPopulationSize;
    bytesize_LearnerActionMatrix        =  sizeof(short) * 1 * learnerPopulationSize ;
    bytesize_PointMatrix                =  sizeof(_point) * pointDim * pointPopulationSize ;
    bytesize_PointLabelMatrix           =  sizeof(int)   * pointPopulationSize  ;



    teamMatrix = (_team*) malloc(bytesize_TeamMatrix);
    teamLengthMatrix = (short*) malloc(bytesize_TeamLengthMatrix);
    teamRewardMatrix = ( _teamReward*) malloc(bytesize_TeamRewardMatrix);
    learnerMatrix = (_learner*) malloc(bytesize_LearnerMatrix);
    learnerBidMatrix = (_learnerBid*) malloc(bytesize_LearnerBidMatrix);
    learnerActionMatrix = (_learnerAction*) malloc(bytesize_LearnerActionMatrix);
    pointMatrix = (_point*) malloc(bytesize_PointMatrix);
    pointLabelMatrix = (_pointLabel*) malloc(bytesize_PointLabelMatrix);


    //learnerMatrix = (_learner*)AllocPinnedMemory( bytesize_LearnerMatrix);
   // pointMatrix = (_point*)AllocPinnedMemory( bytesize_PointMatrix);


    learnerMatrix[0] = 1;
    learnerPool = new FreeMap<int>(0,learnerPopulationSize);
    teamPool = new FreeMap<int>(0,teamPopulationSize);
    pointPool = new FreeMap<int>(0,pointPopulationSize);


    memset(teamMatrix,0,bytesize_TeamMatrix);
    memset(teamLengthMatrix,0,bytesize_TeamLengthMatrix);
    memset(teamRewardMatrix,0,bytesize_TeamRewardMatrix);
    memset(learnerMatrix,0,bytesize_LearnerMatrix);
    memset(learnerBidMatrix,0,bytesize_LearnerBidMatrix);
    memset(learnerActionMatrix,0,bytesize_LearnerActionMatrix);
    memset(pointMatrix,0,bytesize_PointMatrix);
    memset(pointLabelMatrix,0,bytesize_PointLabelMatrix);


    for(int i=0;i < teamPopulationSize;i++)
    {
        teamAge[i] = 0;
    }


}

DataEngine::~DataEngine()
{

    cout << "DTOR" << endl;
    free(teamMatrix);
    free(teamLengthMatrix);
    free(teamRewardMatrix);
    free(learnerMatrix);
    free(learnerBidMatrix);
    free(learnerActionMatrix);
    free(pointMatrix);
    free(pointLabelMatrix);
   // FreePinnedMemory(learnerMatrix);
   // FreePinnedMemory(pointMatrix);

    delete learnerPool;
    delete teamPool;
    delete pointPool;
}

void DataEngine::initPoints()
{
    int numNewPoints = pointPopulationSize ;//- pointGenerationGapSize;
    cout << " CREATE NEW PTS" << pointPopulationSize << "  "<< pointGenerationGapSize << "  " << numNewPoints << " "<< pointPool->size() << endl;

   // set<_pointId> usedIds;

    _pointLabel label=0;

    //map< _pointLabel,long> counts;
    vector<_pointLabel> labels = DS->getLabelVector();

    _pointId pointMatrixIndex =0;

    for (int i = 0; i < numNewPoints; i++)
    {

        do
        {
            label = labels[drand48() * labels.size()];
        } while (usedPointLabelCounts[label] == DS->getTrainingLabelCount(label));

       _pointId pid = DS->getUniformTrainingPointWithLabel(label);

      // Copy Point
       _pointId matrixIndex =  pointPool->getNextFree();
       _point* pm = getPtrToPointMatrix(matrixIndex);
       DS->copyTrainingPointToMatrix(pid, pm);

       pointIndexMap[pid] = matrixIndex;
       indexPointMap[matrixIndex] = pid;
       pointAge[pid] = 0;

       // Copy Label
       pointLabelMatrix[matrixIndex] = label;
       usedPointLabelCounts[label]++;

       newPointIndicies.push_back(matrixIndex);
  //     cout << " NEW Point [ Id: " << pid << "  index: " << matrixIndex << "  label:  " << label << "  ] " << endl;
    }

//   printf("LABEL COUNTS:      ");
//    for (int i=0; i <= labels.size();i++)
//    {
//        printf("        %ld ",usedPointLabelCounts[i]);
//    }
//    printf("\n");
}


void DataEngine::initTeams(long _t)
{
        int action1, action2;
        _learnerIndex lid;
        _teamIndex tid;

        int numNewTeams = teamPopulationSize ;//- teamGenerationGapSize;

        for (int i=0; i < numNewTeams;i++)
        {
            //Get two different Actions
                 action1 = (int)(drand48() * actionCount);
            do { action2 = (int)(drand48() * actionCount);} while (action1 == action2);


        tid = newTeam(_t);

        lid = newLearner(action1);

        addLearnerToTeam(lid,tid);

        lid = newLearner(action2);

        addLearnerToTeam(lid,tid);

        }
}

void DataEngine::addLearnerToTeam(_learnerIndex li, _teamIndex ti)
{

    short teamSize = teamLengthMatrix[ti];
    _team* T = getPtrToTeamMatrix(ti);
    T[teamSize] = li;
    learnerReferenceCounts[li]++;

    teamLengthMatrix[ti]++;

}

void DataEngine::generatePoints(long _t)
{
   // set<_pointId> usedIds;

    _pointLabel label=0;

    //map< _pointLabel,long> counts;
    vector<_pointLabel> labels = DS->getLabelVector();

    _pointId pointMatrixIndex =0;

  //  cout << " CREATE NEW PTS" << pointPopulationSize << "  "<< pointGenerationGapSize << "  "<< pointPool->size() << endl;
    while(pointPool->hasFreeIndex())
    {

        int __i =0;
        do
        {
            label = labels[drand48() * labels.size()] ;
            if( __i++ > 100 ) {cout << "DEAD LOCK" << endl;}
        } while (usedPointLabelCounts[label] == DS->getTrainingLabelCount(label));

       _pointId pid = DS->getUniformTrainingPointWithLabel(label);

      // Copy Point
       _pointId matrixIndex =  pointPool->getNextFree();
       _point* pm = getPtrToPointMatrix(matrixIndex);
       DS->copyTrainingPointToMatrix(pid, pm);

       pointIndexMap[pid] = matrixIndex;
       indexPointMap[matrixIndex] = pid;
       pointAge[pid] = _t;

       // Copy Label
       pointLabelMatrix[matrixIndex] = label;
       usedPointLabelCounts[label]++;
       newPointIndicies.push_back(matrixIndex);
     //  cout << "Action: NEW Point [ Id: " << pid << "  index: " << matrixIndex << "  label:  " << label << "  ] " << endl;
    }

//   printf("LABEL COUNTS:      ");
//    for (int i=0; i <= labels.size();i++)
//    {
//        printf("        %ld ",usedPointLabelCounts[i]);
//    }
//    printf("\n");


  //  printPointLabelMatrix();
}

//void DataEngine::generateTeamsRaw(long _t)
//{
//
//
//}

void DataEngine::generateTeams(long _t)
{
 //   cout << "\n:::::::::::::::::::::::\n  GENERATE TEAMS   \n::::::::::::::::::::\n";
 //   printLearnerMatrix();
 //   printTeamMatrix();
    vector <_teamIndex> parents;
    int size;

   _teamIndex p1,p2;
   _teamIndex c1,c2,cx,cy;

    /* Learner in parents. */
    set < _learnerIndex> plearners1;
    set < _learnerIndex> plearners2;

    /* Learners in children. */
    set < _learnerIndex> clearners1;
    set < _learnerIndex> clearners2;

    set <_learnerIndex> :: iterator leiter, leiterend;

    /* Set intersection and difference. */
    set < _learnerIndex > setinter;
    set < _learnerIndex > setdiff;

    set < _teamIndex> :: iterator teiter, teiterend;


   // cout << "POOLSIZE:  "<<teamPool->usedElements.size() << endl;

    for(teiter = teamPool->usedElements.begin(), teiterend = teamPool->usedElements.end();
        teiter != teiterend; teiter++){

        parents.push_back(*teiter);
        }
    size = parents.size();

 //   printf("Init TeamSizes: %d of %d\n", teamPool->size(), teamPopulationSize);
    while(teamPool->size()+1 < teamPopulationSize)
    {
 //   printTeamMatrix();
        /* Create two children for each pair of parents. */
        p1 = parents[(int) (drand48() * size)];

        do
        {
            p2 = parents[(int) (drand48() * size)];
        } while(p1 == p2);
#ifdef MYDEBUG
    //    cout << "scm:genTeams " << t << " p1 "  << *p1 << " p2 " << *p2;
#endif

      // cout << " @Parents: " << p1 << "   " << p2 << endl;
        plearners1.clear();
        plearners2.clear();


        //plearners1 = p1->getUniqueMembers()  ;
        _team* p1M = getPtrToTeamMatrix(p1);
        int i=0;
        while(p1M[i] !=-1 && i < maxTeamSize)
        {
           plearners1.insert(p1M[i++]);
        }

        i=0;
        _team* p2M = getPtrToTeamMatrix(p2);
        while(p2M[i] !=-1 && i < maxTeamSize)
        {
           plearners2.insert(p2M[i++]);
        }
        //plearners2 = p2->getUniqueMembers()  ;



        setinter.clear();
        setdiff.clear();

        set_intersection(plearners1.begin(), plearners1.end(),
                         plearners2.begin(), plearners2.end(),
                         insert_iterator< set<int> > (setinter, setinter.begin()));

        set_symmetric_difference(plearners1.begin(), plearners1.end(),
                                 plearners2.begin(), plearners2.end(),
                                 insert_iterator< set<int> > (setdiff, setdiff.begin()));


        set<_learnerIndex> s = plearners1;
//        cout << endl << " P1: ";
//        for(leiter = s.begin(); leiter != s.end(); leiter++)
//        {
//                cout << (*leiter) << "  ";
//        }
//         s = plearners2;
//        cout << endl << " P2: ";
//        for(leiter = s.begin(); leiter != s.end(); leiter++)
//        {
//                cout << (*leiter) << "  ";
//        }

//         s = setinter;
//        cout << endl << " I : ";
//        for(leiter = s.begin(); leiter != s.end(); leiter++)
//        {
//                cout << (*leiter) << "  ";
//        }

//         s = setdiff;
//        cout << endl << " D : ";
//        for(leiter = s.begin(); leiter != s.end(); leiter++)
//        {
//                cout << (*leiter) << "  ";
//        }
//        cout << endl;

        c1 = newTeam(_t);
        c2 = newTeam(_t);

     //   cout << " Children: C1: " << c1 << "   C2:  " << c2 << endl;

        for(leiter = setinter.begin(), leiterend = setinter.end();
            leiter != leiterend; leiter++)
        {
           // cout << " Adding Both " << *leiter << endl;
           addLearnerToTeam(*leiter, c1);
           addLearnerToTeam(*leiter, c2);
          //  c1->addLearner(*leiter);
          //  c2->addLearner(*leiter);
        }


        for(leiter = setdiff.begin(), leiterend = setdiff.end();
            leiter != leiterend; leiter++)
        {
            if(drand48() < 0.5)
            { cx = c1; cy = c2; }
            else
            { cx = c2; cy = c1; }

            if((teamLengthMatrix[cx] < maxTeamSize) && ((teamLengthMatrix[cx] >= 2 && teamLengthMatrix[cy] < 2) == 0))
            {
                addLearnerToTeam(*leiter,cx);
        //        cout << " Adding " << *leiter << " to team " << cx << endl;
            }else{
        //        cout << " Adding " << *leiter << " to team " << cy << endl;
                addLearnerToTeam(*leiter,cy);
            }
        }
        if((teamLengthMatrix[c1] > maxTeamSize || teamLengthMatrix[c2] > maxTeamSize) ||
                (teamLengthMatrix[c1] < 2 && teamLengthMatrix[c2] < 2))
            die(__FILE__, __FUNCTION__, __LINE__, "messed up team generation");



        clearners1.clear();
        clearners2.clear();

        //clearners1 = c1->getUniqueMembers();
        //clearners2 = c2->getUniqueMembers();

        _team* c1M = getPtrToTeamMatrix(c1);
        i=0;
        while(c1M[i] !=-1 && i < maxTeamSize)
        {
           clearners1.insert(c1M[i++]);
        }

        _team* c2M = getPtrToTeamMatrix(c2);
        i=0;
        while(c2M[i] !=-1 && i < maxTeamSize)
        {
           clearners2.insert(c2M[i++]);
        }

        set<_learnerIndex>::iterator it;

//        cout << " Plearners1: ";
//        for(it=plearners1.begin(); it != plearners1.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;
//        cout << " Plearners2: ";
//        for(it=plearners2.begin(); it != plearners2.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;

//        cout << " clearners1: ";
//        for(it=clearners1.begin(); it != clearners1.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;
//        cout << " clearners2: ";
//        for(it=clearners2.begin(); it != clearners2.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;


        /* If the offspring has the same learners as a parent, require that the
       offspring is mutated. */

        if(clearners1 == plearners1 || clearners1 == plearners2){

    //        cout << "CL1 Colliosn " << endl;
            while(generateTeams(_t,c1) == 0){}
        }else
            generateTeams(_t,c1);

        if(clearners2 == plearners1 || clearners2 == plearners2){
    //        cout << "CL2 Colliosn " << endl;
            while(generateTeams(_t,c2) == 0){}
        }else
            generateTeams(_t,c2);


//        cout << "AFTER\n;";//  Plearners1: ";
//        for(it=plearners1.begin(); it != plearners1.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;
//        cout << " Plearners2: ";
//        for(it=plearners2.begin(); it != plearners2.end();it++)
//        {
//           cout <<"   " << (*it);
//        }
//        cout << endl;

//        cout << " C1: ";
//        c1M = getPtrToTeamMatrix(c1);
//        i=0;
//        while(c1M[i] !=-1 && i < maxTeamSize)
//        {
//           cout << "   " << c1M[i++];
//        }
//        cout << endl;

//        cout << " C2: ";
//        c2M = getPtrToTeamMatrix(c2);
//        i=0;
//        while(c2M[i] !=-1 && i < maxTeamSize)
//        {
//           cout << "   " << c2M[i++];
//        }
//        cout << endl;

//        cout << " =-=-=-=-=-=-=-=-=-=-=-= " << endl;

//        i=0;
//        cout << "Action:  NEW Team: " << c1 << "[ ";
//        _team* T = getPtrToTeamMatrix(c1);
//        while( i < maxTeamSize && T[i] != -1)
//        {
//            cout << "  " << T[i] ;
//           // learnerReferenceCounts[T[i]]++;
//            i++;
//        }
//        cout << " ]" << endl;

//        i=0;
//        cout << "Action:  NEW Team: " << c2 << "[ ";
//        T= getPtrToTeamMatrix(c2);
//        while( i < maxTeamSize && T[i] != -1)
//        {
//            cout << "  " << T[i] ;
//           // learnerReferenceCounts[T[i]]++;
//            i++;
//        }
//        cout << " ]" << endl;



     //   cout << endl << endl;
    }


 //   cout << "\n:::::::::::::::::::::::\n  END GENERATE TEAMS   \n::::::::::::::::::::\n";

}

void DataEngine::generateTeamsNew(long _t)
{
//    cout << "\n:::::::::::::::::::::::\n  GENERATE TEAMS   \n::::::::::::::::::::\n";


    _teamId ParentA_Index;
    _teamId ParentB_Index;


    set<_learner> PA_learnerSet;
    set<_learner> PB_learnerSet;



    int needMoreTeams = 0;
    while(needMoreTeams)
    {
        //2 SelectParents





    }


    cout << "\n:::::::::::::::::::::::\n  END GENERATE TEAMS   \n::::::::::::::::::::\n";
}

bool DataEngine::generateTeams(long _t,_teamIndex c)
{


    int t = 0;

    /* Whether the team has been changed by this method. */
    bool changedTeam = 0;
    /* If a learners is changed. */
    bool changedLearner =0;

    set < _learnerIndex> lSet;
    vector < _learnerIndex> lVec;
    vector < int> :: iterator veiter, veiterend;

    set < int > :: iterator seiter;
    //  int i;

   // _learnerIndex *lr;
    int left =0;

    //  int index;





    /* Retrieve learners in offspring and insert into vector. */
    _learnerIndex* L = getPtrToTeamMatrix(c);

//    cout << "   Learner(" << c << "):";
//    for(int j =0; j < maxTeamSize; j++)
//    {
//        cout << "\t" << L[j];
//    }
//    cout << endl;


    int i=0;
    while(L[i] != -1 && i < maxTeamSize)
    {
        lSet.insert( L[i++] );
       //learnerReferenceCounts[L[i]]--;
    }

    lVec.insert(lVec.begin(), lSet.begin(), lSet.end());

    /* We want to consider the learners in arbitrary order. */
    random_shuffle(lVec.begin(), lVec.end());
    /* How many learners were present originally. */

    /* Delete learners from team (but keep them in lVec). */

    for(veiter = lVec.begin(), veiterend = lVec.end();
        (veiter != veiterend) && (lSet.size() > 2); veiter++)
    {
        if(drand48() < PROB_LearnerDeletion)
        {
//            cout << "Delete Learner " << *veiter << endl;
            lSet.erase(*veiter);
            learnerReferenceCounts[(*veiter)]--;
            left--;
            changedTeam = 1;
#ifdef MYDEBUG
            cout << " " << (*veiter);
#endif
        }
    }

    for(veiter = lVec.begin(), veiterend = lVec.end();
        (veiter != veiterend) && (lSet.size() < maxTeamSize); veiter++)
    {
        if(drand48() < PROB_LearnerAddition)
        {
            /* Add a mutated learner. */

//            cout << " Enter Add " << *veiter << endl;

            changedLearner = 0;

            _learnerIndex srclid = *veiter;

            vector<_learner> tmp_learner;
            tmp_learner.reserve(20);


            //_learner* tmp_learner = (_learner*) malloc ( sizeof(_learner) * learnerLength);
            _learnerAction tmp_action = learnerActionMatrix[srclid];
            _learner* srcL = getPtrToLearnerMatrix(srclid);

          // cout << "        POTENTIALLY MUTATing LEARNER(" << srclid << " )" << endl;
           // for(int x=0; x <= srcL[0];x++){ tmp_learner[x] = srcL[x];}

 //           cout << "SRC:" << srcL[0] ;
            for(int x=1; x <= srcL[0];x++){
  //               cout << "    " << srcL[x];
                tmp_learner.push_back(srcL[x]);}

  //          cout <<endl;
          //  cout << "TMP: " p<<  tmp_learner[0] << endl;
  //           cout << "PRE MUTATE" << tmp_learner.size() << endl;


 //           cout << "\t\tPRE: ";
//            for(int q=0; q< tmp_learner.size();q++)
//            { cout << "   " << tmp_learner[q];}
//            cout<< endl;




             changedLearner = mutateBid(&tmp_learner);
//            cout << "\t\tPST: ";
 //           for(int q=0; q< tmp_learner.size();q++)
 //           { cout << "   " << tmp_learner[q];}
 //           cout<< endl;
            /* Mutate action. */

                if(drand48() < PROB_ActionMutation)
                {
                    _learnerAction action = (_learnerAction)(drand48() * usedPointLabelCounts.size());
                    if ( action != tmp_action)
                    {
                    	//cout << "New Action " << action << " Old Action: " << tmp_action << endl;
                        tmp_action = action;
                        changedLearner = true;

                    }

                }
            /* Keep only learner if changed. */

                 if(changedLearner == 0)
                    {
                     // delete lr;            // DELETED OUTSIDE OF CONDITIONALp
  //                    cout << "     No Learner Added" << endl;
                    }
                  else
                    {


                      _learnerId lid = newLearner(tmp_action,tmp_learner);

                    //  cout << "     Added Learner (Lid:" << lid << ") " << endl;

                      //c->addLearner(lr);
                      //c->addLearner(lid);
                      lSet.insert(lid);
                      learnerReferenceCounts[lid]++;
                      //_L.insert(lr);
                      changedTeam = 1;

                      left++;

#ifdef MYDEBUG
    //        cout << "scm:genTeams added " << *lr << " from " << **veiter << endl;
#endif
                   }
  //      free(tmp_learner);
    }
}

    i =0;
    _team* T = getPtrToTeamMatrix(c);
    vector<_learnerIndex> v(lSet.size());
    std::copy(lSet.begin(), lSet.end(), v.begin());
    random_shuffle(v.begin(),v.end());
    vector<_learnerIndex>::iterator it = v.begin();
    for(;it != v.end();i++)
    {
      //  cout << "  " << (*it);
        T[i] =  (*it);
        //learnerReferenceCounts[(*it)]++;
        it++;
    }
    for (;i < maxTeamSize;i++)
    {
        T[i] = -1;
    }
    teamLengthMatrix[c] = v.size();


// THE FOLLOWING SHOULD NOT HAPPEN!!!

/* Make sure don't get teams with fewer than two learners. */

while(lSet.size() < 2)
{
    cout << " ERRRROROOROROROOROR " << endl;

    /* Pick a random learner in the population. */
    int index = (int) (drand48() * learnerPool->size());


    /* Advance to that learner. */
    int i;

    /* Try to add the learner .*/

        addLearnerToTeam(index, c);
        printf("NEW LEARNER(%d) ADDED TO TEAM(%d)\n", index,c);

#ifdef MYDEBUG
    cout << "scm:genTeams small " << endl; // << c->size() << " added " << **seiter << endl;
#endif
}

if(teamLengthMatrix[c] < 2)
die(__FILE__, __FUNCTION__, __LINE__, "c->size() < 2");

return changedTeam;
//return true;
}

void DataEngine::evaluate()
{
    //EvaluateLearners(4,2);
}

bool DataEngine::initDataSets(map < string,string> &args)
{
    string trainSetName, testSetName;
    long trainSetSize, testSetSize;
    int setDim;
    string envType;
    long _numActions;

    map < string, string > :: iterator maiter;

    if((maiter = args.find("trainSetName")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg trainSetName");

    trainSetName = maiter->second;

    if((maiter = args.find("testSetName")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg testSetName");

    testSetName = maiter->second;

    if((maiter = args.find("trainSetSize")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg trainSetSize");

    trainSetSize = stringToLong(maiter->second);

    if((maiter = args.find("testSetSize")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg testSetSize");

    testSetSize = stringToLong(maiter->second);

    if((maiter = args.find("setDim")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg setDim");

    setDim = stringToInt(maiter->second);
    pointDim = setDim;
    if((maiter = args.find("envType")) == args.end())
      die(__FILE__, __FUNCTION__, __LINE__, "cannot find arg envType");

    envType = maiter->second;

    cout << "arg trainSetName " << trainSetName << endl;
    cout << "arg testSetName " << testSetName << endl;
    cout << "arg trainSetSize " << trainSetSize << endl;
    cout << "arg testSetSize " << testSetSize << endl;
    cout << "arg setDim " << setDim << endl;
    cout << "arg envType " << envType << endl;

    //if(setDim < 1 || setDim > INPUTS)
    if(setDim < 1 || setDim > 64)
      die(__FILE__, __FUNCTION__, __LINE__, "bad arg setDim < 1 || setDim > INPUTS");

    if(trainSetSize < 1 || testSetSize < 1)
      die(__FILE__, __FUNCTION__, __LINE__, "bad arg setSize < 1 || testSetSize < 1");



            //////////////////////////////////////////////
           //////////////////////////////////////////////
          //////////////////////////////////////////////


    if(envType == "datasetEnv")
      {
        DS = new Dataset(setDim);
        DS->initTrainingSet(trainSetName, trainSetSize);
        DS->initTestingSet(testSetName, testSetSize);
        cout << "arg envType datasetEnv" << endl;
      }
    else
      {
        die(__FILE__, __FUNCTION__, __LINE__, "bad arg envType");
      }

    _numActions = DS->numLabels();

    if(_numActions > maxTeamSize)
      die(__FILE__, __FUNCTION__, __LINE__, "numActions > omega");

    // Moved into InitTestingSet
   // if(DS->numLabels() != DS_test->numLabels())
   //   cout << "UNHANDLED!! WARNING different number of labels in training and test sets" << endl;

   // for(int i = 0; i < _numActions; i++)
   //   if(DS_train->getLabel(i) != DS_test->getLabel(i))
   //     die(__FILE__, __FUNCTION__, __LINE__, "labels do not match");

        map<_pointLabel,vector<_pointId>* >::iterator trainLabel_it = DS->trainingLabelMap.begin();
        map<_pointLabel,vector<_pointId>* >::iterator trainLabel_itend = DS->trainingLabelMap.end();
        map<_pointLabel,vector<_pointId>* >::iterator testLabel_it = DS->testingLabelMap.begin();

        while (trainLabel_it != trainLabel_itend)
        {
           if  ((*testLabel_it).first != (*trainLabel_it).first)
             die(__FILE__, __FUNCTION__, __LINE__, "labels do not match");

           trainLabel_it++;
            testLabel_it++;
        }

    cout << "arg numActions train/test " << _numActions << "/" << DS->numLabels() << endl;

    if(_numActions < 2)
      die(__FILE__, __FUNCTION__, __LINE__, "bad arg numActions < 2");

    return true;
}
_teamId DataEngine::newTeam(long _t)
{
    if (false == teamPool->hasFreeIndex())
        die(__FILE__, __FUNCTION__, __LINE__, "No Room for Team");

    int tindex = teamPool->getNextFree();
    _team* T =  getPtrToTeamMatrix(tindex);

    // Zero Team
    for(int i=maxTeamSize-1;i >=0; i--){T[i] = -1;}

    teamLengthMatrix[tindex] = 0;
    teamAge[tindex] = _t;
    return tindex;
}

_learnerIndex DataEngine::newLearner(_learnerAction action)
{

    int progSize = 1+  ((int) (drand48() * maxProgSize));


    if (false == learnerPool->hasFreeIndex())
        die(__FILE__, __FUNCTION__, __LINE__, "No Room for learner");

    _learnerIndex index = learnerPool->getNextFree();

    _learner* L = getPtrToLearnerMatrix(index);

    for(int i=0;i < learnerLength; i++)
    {
        L[i] = 0;
    }


    L[0] = progSize;
    for(int i=1; i< progSize+1;i++)
    {
        _instruction in;
        for(int _j = 0 ; _j <in.size(); _j++)
            if(drand48()>0.5) in.flip(_j) ;
        L[i] = (_learner)in.to_ulong();
    }for(int i= progSize+1; i< learnerLength;i++)
    {
        L[i] = EMPTY_INSTRUCTION;

    }
//    for(int i=1; i< progSize+1;i++)
//    {
//        for(int _j = 0 ; _j <13; _j++)
//            L[i] += (drand48()>0.5) << _j;
//       // L[i] = rand() % 32767;
//    }for(int i= progSize+1; i< learnerLength;i++)
//    {
//        L[i] = EMPTY_INSTRUCTION;

//    }

    newLearnerIndicies.push_back(index);
    learnerActionMatrix[index] = action;

    return index;
}

_learnerIndex DataEngine::newLearner(_learnerAction action, _learner* Src)
{


    if (false == learnerPool->hasFreeIndex())
        die(__FILE__, __FUNCTION__, __LINE__, "No Room for learner");

    _learnerIndex index = learnerPool->getNextFree();
    //_learnerId lid = nextLearnerId();

    _learner* L = getPtrToLearnerMatrix(index);
    for(int i=0;i < learnerLength; i++)
    {
        L[i] = 0;
    }
    for( int i=0; i < learnerLength; i++)
    {
       L[i] = Src[i];
    }

    learnerActionMatrix[index] = action;
    //cout << "NewLearner2:" << action << endl;
    //learnerToIndexMap[lid] = index;
    //indexToLearnerMap[index] = lid;
    newLearnerIndicies.push_back(index);
    return index;
}
_learnerIndex DataEngine::newLearner(_learnerAction action, vector<_learner> &Src)
{


    if (false == learnerPool->hasFreeIndex())
        die(__FILE__, __FUNCTION__, __LINE__, "No Room for learner");

    _learnerIndex index = learnerPool->getNextFree();
    //_learnerId lid = nextLearnerId();

    _learner* L = getPtrToLearnerMatrix(index);


    L[0] = Src.size();
    int i;
    for( i=1; i < Src.size()+1; i++)
    {
       L[i] = Src[i-1];
    }
    for( ;i< learnerLength; i++)
    {
        L[i] = EMPTY_INSTRUCTION;
    }

    learnerActionMatrix[index] = action;
 //   cout << "NewLearner3:" << action << endl;
    //learnerToIndexMap[lid] = index;
    //indexToLearnerMap[index] = lid;
    newLearnerIndicies.push_back(index);
    return index;
}


bool DataEngine::mutateBid(vector<_learner>* vec)
{
    bool changed = false;


//    cout << "\t\tMut: ";
//    for(int x= 0; x < vec->size() ;x++){ cout << "   " << (*vec)[x];}
//    cout << endl;


    // REmove Random Instruction
    if( vec->size()> 1 && drand48() < PROB_BidDeletion)
    {
        int i = (int) (drand48() * vec->size());

        vec->erase(vec->begin()+i);
        changed = true;

    //    cout << "\t\t\tlearner:muBid delete " << i << endl;
    }

    /* Insert random instruction. */
    if( vec->size() < maxProgSize && drand48() < PROB_BidAddition)
    {
        _learner instr = 0;
        for( int _j=0; _j <13;_j++)
        {
            instr += (drand48() > 0.5) << _j;
        }

        int i = (int) (drand48() * ( vec->size() + 1)) ;

        if ( i < vec->size()){
            vec->insert(vec->begin()+i,instr);
        }else{
            vec->push_back(instr);
        }
        changed = true;

     //   cout << "\t\t\tlearner:muBid add " << i << " "<< instr << endl;
    }


    /* Flip single bit of random instruction. */
    if(drand48() < PROB_BidMutate)
    {
        int i = (int) (drand48() * vec->size());

        _instruction in((*vec)[i]);

        int j = (int) (drand48() * in.size() ); // eight bits per byte
        in.flip(j);

        (*vec)[i] = (short)in.to_ulong();

        changed = true;

        //#ifdef MYDEBUG
     //   cout << "\t\t\tlearner:muBid mutate " << i << ", " << j << "VEC.size()" << vec->size()<< endl;
        //#endif
    }

    /* Swap positions of two instructions. */
    if(vec->size() > 1 && drand48() < PROB_BidSwap)
    {

        int i;

        do
        {
            i= (int) (drand48() * vec->size());
            //       cout << " I=" << i << "VS:" << vec->size() << endl;
        } while( i==0);

        int j;
        do
        {
            j = (int) (drand48() * vec->size());
        } while(i == j );
        //      printf("BIDSIZE: %d    I:%d    J:%d Vs:%d ", vec->size(), i  , j, vec->size());
        // for( int _i=0; _i < vec->size(); _i++)
        //     printf( "  %d:%d ", _i, vec->at(_i));
        // printf("\n");
        _learner tmp = vec->at(i);
        vec->at(i) =vec->at(j);
        vec->at(j) =tmp;

        changed = true;

#ifdef MYDEBUG
     //   cout << "\t\t\tlearner:muBid swap " << i << ", " << j << endl;
#endif
    }

//    cout << "\t\tEnd: ";
//    for(int x= 0; x < vec->size() ;x++){ cout << "   " << (*vec)[x];}
//    cout << endl;
    return changed;

}






_learner* DataEngine::getPtrToLearnerMatrix(int index)
{
    if (index > learnerPopulationSize)
        die(__FILE__, __FUNCTION__, __LINE__, "learnerIndex Out of Range");

    return &learnerMatrix[index * learnerLength];
}


_team* DataEngine::getPtrToTeamMatrix(int index)
{
    if (index > teamPopulationSize)
        die(__FILE__, __FUNCTION__, __LINE__, "teamIndex Out of Range");
    return &teamMatrix[index * maxTeamSize];
}

_point* DataEngine::getPtrToPointMatrix(int index)
{
    if (index > pointPopulationSize)
        die(__FILE__, __FUNCTION__, __LINE__, "pointIndex Out of Range");
    return &pointMatrix[index * pointDim];
}





_learnerId DataEngine::duplicateLearner(_learnerId lid)
{
   _learnerId newId= learnerPool->getNextFree();
   _learner* newLearner = getPtrToLearnerMatrix(newId);
   _learner* oldLearner = getPtrToLearnerMatrix(lid);

   for(int i=0; i < learnerLength; i++)
   {
        newLearner[i] = oldLearner[i];
   }
   newLearnerIndicies.push_back(newId);
   return newId;
}










void DataEngine::printTeamMatrix()
{
   cout << "==========================\n Team Matrix \n====================\n";

    for(int i=0; i < teamPool->maxSize(); i++)
    {
        cout << i << ")   ";
        for(int j = 0; j < maxTeamSize; j++)
        {
            std::cout << "  " <<teamMatrix[i*maxTeamSize + j];
        }
        std::cout << std::endl;
    }

}
void DataEngine::printPossibleTeamActions()
{
   cout << "==========================\n Team Actions \n====================\n";

    for(int i=0; i < teamPool->maxSize(); i++)
    {
        cout << i << ")   ";
        for(int j = 0; j < maxTeamSize; j++)
        {
            if ( 0 > teamMatrix[i*maxTeamSize + j])
            { break;}
            std::cout << " - " <<learnerActionMatrix[teamMatrix[i*maxTeamSize + j]];
        }
        std::cout << std::endl;
    }

}

void DataEngine::printLearnerMatrix()
{
   cout << "==========================\n Learner Matrix \n====================\n";
   cout << "  LeanerLength: "<<learnerLength << "   LearnerCount: "<< learnerPool->size() << endl;
    for(int i=0; i < learnerPool->maxSize(); i++)
    {
        cout << i << ")   ";
     //   cout << "L("<<indexToLearnerMap[i] << ")::   ";
        for(int j = 0; j < learnerLength; j++)
        {
            float p = 3;
            float n = learnerMatrix[i*learnerLength + j];
            std::cout << " " << floor(pow(10, p) * n) / pow(10, p);
        }
        //std::cout << " ## " <<  getLearnerAction(i);
        cout <<  std::endl;
    }

}

void DataEngine::printLearnerActionMatrix()
{
   cout << "==========================\n Learner Action Matrix \n ==============\n";
    for(int i=0; i < learnerPool->maxSize(); i++)
    {
        cout << i << ")   ";
     //   cout << "L("<<indexToLearnerMap[i] << ")::   ";
            std::cout << " " << getLearnerAction(i);
        //std::cout << " ## " <<  getLearnerAction(i);
        cout <<  std::endl;
    }

}

void DataEngine::printPointMatrix()
{
   cout << "==========================\n Point Matrix \n====================\n";
//   cout << "  PointDim: "<< << "   LearnerCount: "<< maxLearnerCount << endl;

   int pointLength = DS->dim;
   for(int i=0; i < pointPool->maxSize(); i++)
    {
        cout << i << ")   ";
        for(int j = 0; j < pointLength; j++)
        {
            std::cout << " " <<pointMatrix[i*pointLength + j];
        }
        //std::cout << " :::: " << pointLabelMatrix[i];
        cout << std::endl;
    }

}

void DataEngine::printPointLabelMatrix()
{
   cout << "==========================\n Point Label Matrix \n====================\n";

   for(int i=0; i < pointPool->maxSize(); i++)
    {
        cout << i << ")   ";
            std::cout << " -  " <<pointLabelMatrix[i] << endl;
    }


}


void DataEngine::printBidMatrix()
{
   cout << "==========================\n Learner Bid Matrix \n====================\n";
   cout << "  PointDim: "<< pointDim << "   LearnerCount: "<<  learnerPool->size() << endl;

   for(int i=0; i < learnerPool->size(); i++)
    {
        cout << i << ")   ";
      //  cout.setf(ios::fixed,ios::floatfield);
        for(int j = 0; j < pointPool->size(); j++)
        {
            float p = 3;
            float n = learnerBidMatrix[i*pointPool->size() + j];
            std::cout << " " << floor(pow(10, p) * n) / pow(10, p);
        }
        cout << endl;
    }

}

void DataEngine::printTeamRewardMatrix()
{
	if(false)
	{

   cout << "==========================\n Team Reward Matrix \n====================\n";
   cout << teamPool->maxSize() << "   " << pointPool -> maxSize() << endl;

    for(int i=0; i < teamPool->maxSize(); i++)
    {
        cout << i << ")   ";
        for(int j = 0; j < pointPool->size(); j++)
        {
             std::cout << " -  " <<teamRewardMatrix[i*pointPool->size() + j];
        }
        cout << endl;
    }

   cout << teamPool->maxSize() << "   " << pointPool -> maxSize() << endl;

	}else{

		  for(int i=0; i < teamPool->maxSize(); i++)
		  {
		        for(int j = 0; j < pointPool->size(); j++)
		        {
		             std::cout << " " <<teamRewardMatrix[i*pointPool->size() + j];
		        }
		        cout << endl;
		    }

	}

}

void DataEngine::printLearnerReferenceCounts()
{
    cout << "==========================\n Learner Reference counts \n====================\n";

     for(int i=0; i < learnerPool->maxSize(); i++)
     {
        cout << i << ")   ";
         std::cout << "   " << i << " ::   " << learnerReferenceCounts[i];
         cout << endl;
     }

}
