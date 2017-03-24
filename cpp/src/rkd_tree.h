//
//  RKD_tree.h
//  Created by Zhen Zhai on 5/15/15.
//  Copyright (c) 2015 Zhen Zhai. All rights reserved.
//

#ifndef RKD_TREE_H
#define RKD_TREE_H

#include "kd_tree.h"

using namespace std;

/* Class Prototypes */

template<class Label, class T>
class RKDTree;

/*
 * Name             : RKDTree
 * Description      : Encapsulates the KDTreeNodes into multiple trees.
 * Data Field(s)    : None.
 * Function(s)      : RKDTree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    RKDTree(size_t, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size and
 *                            data set
 *                    RKDTree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization
 */
template<class Label, class T>
class RKDTree : public KDTree<Label, T>
{
private:
    static KDTreeNode<Label, T> * build_tree(size_t min_leaf_size,
                                             DataSet<Label, T> & st, vector<size_t> domain);
    RKDTree(DataSet<Label, T> & st);
public:
    RKDTree(size_t min_leaf_size, DataSet<Label, T> & st);
    RKDTree(ifstream & in, DataSet<Label, T> & st);
};

/* Private Functions */

template<class Label, class T>
KDTreeNode<Label, T> * RKDTree<Label, T>::build_tree(size_t min_leaf_size,
                                                    DataSet<Label, T> & st, vector<size_t> domain)
{
    LOG_FINE("Enter build_tree\n");
    LOG_FINE("with min_leaf_size = %ld and domain.size = %ld\n", min_leaf_size, domain.size());
    if (domain.size() < min_leaf_size) {
        LOG_FINE("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new KDTreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);
    size_t mx_var_index = ran_variance_index(subst);
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++) {
        values.push_back((*subst[i])[mx_var_index]);
    }
    double pivot = selector(values, (size_t)(values.size() * 0.5));
    vector<size_t> subdomain_l;
    size_t subdomain_l_lim = (size_t)(values.size() * 0.5);
    LOG_FINE("> l_lim = %ld\n", subdomain_l_lim);
    vector<size_t> subdomain_r;
    vector<size_t> pivot_pool;
    for (size_t i = 0; i < domain.size(); i++) {
        if (values[i] == pivot) {
            pivot_pool.push_back(domain[i]);
        } else {
            if (values[i] < pivot)
                subdomain_l.push_back(domain[i]);
            else
                subdomain_r.push_back(domain[i]);
        }
    }
    
    //distribute pivot pool to all the children nodes
    //dot a random vector then do split again
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    
    //extract the vectors from dataset
    DataSet<Label, T> tie_vectors = st.subset(pivot_pool);
    
    //update pool using randome tie breaker
    vector<double> update_pool;
    for (int j = 0; j < tie_vectors.size(); j++) {
        double product = dot(*tie_vectors[j], tie_breaker);
        update_pool.push_back(product);
    }
    
    //find the tie_pivots and distribute tie vectors
    size_t k = subdomain_l_lim - subdomain_l.size();
    
    
    //store new pivots in update_pool
    double tie_pivot;
    
    tie_pivot = selector(update_pool, k);
    for (int j = 0; j < update_pool.size(); j++) {
        if (update_pool[j] <= tie_pivot)
            subdomain_l.push_back(pivot_pool[j]);
        else
            subdomain_r.push_back(pivot_pool[j]);
    }
    
    
    KDTreeNode<Label, T> * result = new KDTreeNode<Label, T> (mx_var_index, pivot, domain, dimension, tie_pivot, tie_breaker);
    result->set_left(build_tree(min_leaf_size, st, subdomain_l));
    result->set_right(build_tree(min_leaf_size, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_FINE("Exit build_tree\n");
    return result;
}

template<class Label, class T>
RKDTree<Label, T>::RKDTree(DataSet<Label, T> & st) :
    KDTree<Label, T>(st)
{
    LOG_INFO("RKDTree Constructed\n");
    LOG_FINE("with default constructor\n");
}

/* Public Functions */

template<class Label, class T>
RKDTree<Label, T>::RKDTree(size_t min_leaf_size, DataSet<Label, T> & st) :
    KDTree<Label, T>(st)
{
    LOG_INFO("RKDTree Constructed\n");
    LOG_FINE("with c = %ld", min_leaf_size);
    this->set_root(build_tree(min_leaf_size, st, st.get_domain()));
}

template<class Label, class T>
RKDTree<Label, T>::RKDTree(ifstream & in, DataSet<Label, T> & st) :
    KDTree<Label, T>(in, st)
{
    LOG_INFO("RKDTree Constructed\n");
    LOG_FINE("with input stream\n");
}

#endif
