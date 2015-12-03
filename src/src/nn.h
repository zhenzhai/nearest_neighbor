/* 
 * File             : nn.h
 * Date             : 2014-5-29
 * Summary          : Basic nearest neighbor algorithms.
 */
#ifndef NN_H_
#define NN_H_

#include <map>
#include <vector>
#include "data_set.h"
using namespace std;

/*
 * Name             : nearest_neighbor
 * Prototype        : vector<T> * nearest_neighbor(const vector<T> *, const DataSet<Label, T> &)
 * Description      : Gets the nearest neighbor to a query in a linear fashion.
 * Parameter(s)     : query     - The vector to search the data set against
 *                    st        - The set to search from
 * Return Value     : Gets the nearest neighbor to the query in the data set
 */
template<class Label, class T>
vector<T> * nearest_neighbor(const vector<T> * query, const DataSet<Label, T> & st)
{
    LOG_INFO("Enter nearest_neighbor\n");
    vector<T> * mn_vtr = NULL;
    double mn_dist = 0;
    double l_dist = 0;
    for (size_t i = 0; i < st.size(); i++) {
        l_dist = distance_to(query, st[i]);
        if (!mn_vtr || l_dist < mn_dist) {
            mn_dist = l_dist;
            mn_vtr = st[i];
        }
    }
    LOG_INFO("Exit nearest_neighbor\n");
    return mn_vtr;
}


/*
 * Name             : k_nearest_neighbor
 * Prototype        : DataSet<Label, T> k_nearest_neighbor(size_t, const vector<T> *, 
 *                                                         const DataSet<Label, T> &)
 * Description      : Gets the k nearest neighbors to a query in a linear fashion.
 * Parameter(s)     : query     - The vector to search the data set against
 *                    st        - The set to search from
 * Return Value     : Gets a data set storing the k nearest neighbors to the query in the data set
 */
template<class Label, class T>
DataSet<Label, T> k_nearest_neighbor(size_t k, vector<T> * query, DataSet<Label, T> & st)
{
    LOG_INFO("Enter k_nearest_neighbor\n");
    map<vector<T> *, double> dist_mp; 
    vector<double> dist_vtr;
    for (size_t i = 0; i < st.size(); i++) {
        double dist = distance_to(query, st[i]);
        dist_mp[st[i]] = dist;
        dist_vtr.push_back(dist);
    }
    double k_dist = selector(dist_vtr, k);
    vector<size_t> domain;
    for (size_t i = 0; domain.size() < k && i < st.size(); i++) {
        if (dist_mp[st[i]] <= k_dist) {
            vector<size_t>::iterator itr;
            for (itr = domain.begin(); itr != domain.end() && dist_mp[st[*itr]] < dist_mp[st[i]]; itr++);
            if (itr != domain.end())
                domain.insert(itr, i);
            else
                domain.push_back(i);
        }
    }
    LOG_INFO("Exit k_nearest_neighbor\n");
    return st.subset(domain);
}

/*
 * Name             : c_approx_nn
 * Prototype        : c_approx(size_tconst vector<T> *, const DataSet<Label, T> &)
 * Description      : Gets the c approxiate nearest neighbors to a query in a linear fashion.
 * Parameter(s)     : c         - The c approximation
 *                    query     - The vector to search the data set against
 *                    st        - The set to search from
 * Return Value     : Gets a data set storing the c approximate nearest neighbors to the query 
 *                    in the data set
 */
template<class Label, class T>
DataSet<Label, T> c_approx_nn(double c, vector<T> * query, DataSet<Label, T> & st, 
        vector<T> * nn)
{
    LOG_INFO("Enter c_approx_nn\n");
    map<vector<T> * , double> dist_mp;
    for (size_t i = 0; i < st.size(); i++) {
        double dist = distance_to(query, st[i]);
        dist_mp[st[i]] = dist;
    }
    double dist = distance_to(query, nn);
    double c_distance = c * dist;
    vector<size_t> domain;
    for (size_t i = 0; i < st.size(); i++) {
        if (dist_mp[st[i]] <= c_distance) {
            domain.push_back(i);
        }
    }
    DataSet<Label, T> & c_approx = st.subset(domain);
    LOG_INFO("Exit c_approx_nn\n");
    return c_approx;
}

#endif
