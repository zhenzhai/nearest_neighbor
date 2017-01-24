#ifndef VECTOR_MATH_H_
#define VECTOR_MATH_H_

#include <vector>
#include <random>

using namespace std;

/* Calculate distance between two vectors */
template<class T>
double distance_to(const vector<T> * v1, const vector<T> * v2)
{
    if (v1->size() != v2->size())
        return -1;
    double distance = 0;
    for (size_t i = 0; i < v1->size(); i++)
    {
        double d = (double)(*v2)[i] - (double)(*v1)[i];
        distance += d * d;
    }
    return distance;
}

/* Calculate dot product of two vectors */
template<class A, class B>
double dot(vector<A> v, vector<B> vd)
{
    long double factor = 0;
    for (int i = 0; i < v.size() && i < vd.size(); i++)
        factor += v[i] * vd[i];
    return factor;
}

/* Find the k smallest value in vector */
template<class T>
T selector(vector<T> st, size_t k)
{
	srand(int(time(NULL)));
	size_t sz = st.size();
	double randomIndex = rand() % sz;
    
	vector<T> left;
	vector<T> right;
	vector<T> v;
    
    typename vector<T>::iterator itr;
	for (itr = st.begin(); itr != st.end(); itr++)
	{
		if(*itr == st[randomIndex])
			v.push_back(*itr);
		else if(*itr < st[randomIndex])
			left.push_back(*itr);
		else
			right.push_back(*itr);
	}
    
	if (left.size() >= k)
		return selector(left, k);
    else if (left.size() + v.size() >= k){
        return st[randomIndex];}
	else
		return selector(right, (size_t)(k - left.size() - v.size()));
}

/* Generate a random tie breaker vector */
vector<double> random_tie_breaker(size_t dimension)
{
    random_device rd;
    default_random_engine generator(rd());
    normal_distribution<double> distribution(0.0,1.0);
    vector<double> tie_breaker(dimension);
    for (int i=0; i<dimension; i++)
        tie_breaker[i] = distribution(generator);
    return tie_breaker;
}



/* Generate a vector by taking difference between two random points*/
template<class T>
vector<double> random_diff(size_t dimension, vector<T> vector_i, vector<T> vector_j)
{
    vector<double> tie_breaker(dimension);
    for (int i=0; i<dimension; i++)
        tie_breaker[i] = vector_i[i] - vector_j[i];
    return tie_breaker;
}

#endif
