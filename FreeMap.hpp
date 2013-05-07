
#ifndef _FREEMAP_H_
#define _FREEMAP_H_

#include <map>
#include <set>
#include <cassert>

template <typename K> 
class FreeMap
{

    public:
    std::set<K> freeElements ;
        std::set<K> usedElements ;



//	FreeMap();
	FreeMap(K from,K to);
	bool flag(K);
	bool free(K);
	bool operator[] (const K& key); 
	bool hasFreeIndex();
	int size();
    K getNextFree();


    int maxSize();
    int maxUsedIndex();

    typename std::set<K>::iterator begin();
    typename std::set<K>::iterator end();


    private:
    int maxsize;
    K maxKey;
    K minKey;

};


#include "FreeMap.cpp"
#endif


