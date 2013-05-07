
#include "FreeMap.hpp"
#include <iostream>


//template <typename K> 
//FreeMap<K>::FreeMap()
//{

//}
template <typename K> 
FreeMap<K>::FreeMap(K from, K to)
{

	K curr = from;
	while (curr != to)
	{
        freeElements.insert(curr);
		curr++;
	}

        maxsize = to;			// BUG
        maxKey = to;
        minKey = from;
}


template <typename K> 
bool FreeMap<K>::flag(K key)
{
	assert(key <= maxKey);
	assert(key >= minKey);
    bool v =  freeElements.count(key);
    freeElements.erase(key);
    usedElements.insert(key);
	
    return v;
}

template <typename K> 
bool FreeMap<K>::free(K key)
{
	assert(key <= maxKey);
	assert(key >= minKey);
    bool curr = usedElements.count(key);
    freeElements.insert(key);
    usedElements.erase(key);


    return curr;
}

template <typename K> 
bool FreeMap<K>::operator[] (const K& key)
{
    return ;
}

template <typename K> 
K FreeMap<K>::getNextFree()
{
	if (freeElements.empty()){
		std::cout << "FreeMap: Invalid Request - No free Indicies ";
		return -1;
	}
	K k = *(freeElements.begin());	
	FreeMap::flag(k);
	return k;
}


template <typename K> 
bool FreeMap<K>::hasFreeIndex()
{
	return (!freeElements.empty());
}

template <typename K> 
int FreeMap<K>::size()
{
    return usedElements.size();
}

template <typename K>
int FreeMap<K>::maxSize()
{
    return maxsize ;
}

template <typename K>
int FreeMap<K>::maxUsedIndex()
{
    return (*usedElements.rbegin()) ;
}

template <typename K>
typename std::set<K>::iterator FreeMap<K>::begin()
{
    return usedElements.begin();
}
template <typename K>
typename std::set<K>::iterator FreeMap<K>::end()
{
    return usedElements.end();
}

/*int main()
{

	FreeMap<int> F(0,10);
	std::cout << F[4] << std::endl;
	std::cout << F.flag(4) << std::endl;
	std::cout << F[4] << std::endl;

}
*/
