#include "Misc.hpp"

void die(const char *file,
     const char *func,
     const int line,
     const char *msg)
{
  cerr << "error in file " << string(file) << " function " << string(func);
  cerr << " line " << line << ": " << string(msg) << "... exiting" << endl;
  abort();
}


int stringToInt(string s)
{
  istringstream buffer(s);

  int i;

  buffer >> i;

  return i;
}

float stringToFloat(string s)
{
  istringstream buffer(s);

  float i;

  buffer >> i;

  return i;
}

long stringToLong(string s)
{
  istringstream buffer(s);

  long i;

  buffer >> i;

  return i;
}


/* Read Arguments File into Map*/
int readMap(string fileName,
            map < string, string > &args)
{
    int pairs = 0;

    ifstream infile(fileName.c_str(), ios::in);

    if(infile == 0)
        die(__FILE__, __FUNCTION__, __LINE__, "cannot open map file");

    do
    {
        string key, value;

        if(infile) infile >> key; else break;
        if(infile) infile >> value; else break;

        args.insert(map < string, string > :: value_type(key, value));
        pairs++;

    } while(true);

    infile.close();

    return pairs;
}
