//
//  multi_kd_tree.h
//  Created by Zhen Zhai on 5/15/15.
//  Copyright (c) 2015 Zhen Zhai. All rights reserved.
//

#ifndef MULTI_KD_TREE_H
#define MULTI_KD_TREE_H

#include "kd_tree.h"

using namespace std;

/* Class Prototypes */

template<class Label, class T>
class MultiKDTree;

/*
 * Name             : MultiKDTree
 * Description      : Encapsulates the KDTreeNodes into multiple trees.
 * Data Field(s)    : None.
 * Function(s)      : MultiKDTree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    MultiKDTree(size_t, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size and
 *                            data set
 *                    MultiKDTree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization
 */
template<class Label, class T>
class MultiKDTree : public KDTree<Label, T>
{
private:
    static KDTreeNode<Label, T> * build_tree(size_t c,
                                             DataSet<Label, T> & st, vector<size_t> domain);
    MultiKDTree(DataSet<Label, T> & st);
public:
    MultiKDTree(size_t c, DataSet<Label, T> & st);
    MultiKDTree(ifstream & in, DataSet<Label, T> & st);
};

/* Private Functions */

template<class Label, class T>
KDTreeNode<Label, T> * MultiKDTree<Label, T>::build_tree(size_t c,
                                                    DataSet<Label, T> & st, vector<size_t> domain)
{
    LOG_INFO("Enter build_tree\n");
    LOG_FINE("with c = %ld and domain.size = %ld\n", c, domain.size());
    if (domain.size() < c) {
        LOG_INFO("Exit build_tree");
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
        if (pivot == values[i]) {
            pivot_pool.push_back(domain[i]);
        } else {
            if (values[i] <= pivot)
                subdomain_l.push_back(domain[i]);
            else
                subdomain_r.push_back(domain[i]);
        }
    }
    
    //distribute pivot pool to all the children nodes
    //dot a random vector then do split again
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    
    //store new pivots in update_pool
    double tie_pivot;
    
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
    tie_pivot = selector(update_pool, k);
    for (int j = 0; j < update_pool.size(); j++) {
        if (update_pool[j] <= tie_pivot)
            subdomain_l.push_back(pivot_pool[j]);
        else
            subdomain_r.push_back(pivot_pool[j]);
    }
    
    
    KDTreeNode<Label, T> * result = new KDTreeNode<Label, T> (mx_var_index, pivot, domain, dimension, tie_pivot, tie_breaker);
    result->set_left(build_tree(c, st, subdomain_l));
    result->set_right(build_tree(c, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_INFO("Exit build_tree\n");
    return result;
}

template<class Label, class T>
MultiKDTree<Label, T>::MultiKDTree(DataSet<Label, T> & st) :
    KDTree<Label, T>(st)
{
    LOG_INFO("MultiKDTree Constructed\n");
    LOG_FINE("with default constructor\n");
}

/* Public Functions */

template<class Label, class T>
MultiKDTree<Label, T>::MultiKDTree(size_t c, DataSet<Label, T> & st) :
    KDTree<Label, T>(st)
{
    LOG_INFO("MultiKDTree Constructed\n");
    LOG_FINE("with c = %ld", c);
    this->set_root(build_tree(c, st, st.get_domain()));
}

template<class Label, class T>
MultiKDTree<Label, T>::MultiKDTree(ifstream & in, DataSet<Label, T> & st) :
    KDTree<Label, T>(in, st)
{
    LOG_INFO("MultiKDTree Constructed\n");
    LOG_FINE("with input stream\n");
}

#endif
