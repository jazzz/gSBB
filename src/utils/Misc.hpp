#ifndef MISC_HPP
#define MISC_HPP

#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <cstdlib>
#include <sys/resource.h>

using namespace std;


void die(const char*, const char *, const int, const char *);


int stringToInt(string);
long stringToLong(string);
float stringToFloat(string);
int readMap(string, map < string, string > &);


template<template <typename> class P = std::less >
struct compare_pair_second {
    template<class T1, class T2> bool operator()(const std::pair<T1, T2>& left, const std::pair<T1, T2>& right) {
        return P<T2>()(left.second, right.second);
    }
};
#endif // MISC_HPP
