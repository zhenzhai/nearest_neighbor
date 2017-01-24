/* 
 * File             : rp_tree.h
 * Summary          : Infrastructure to hold a binary space partition tree.
 */
#ifndef RP_TREE_H_
#define RP_TREE_H_

#include "vector_math.h"
#include "pca_tree.h"
using namespace std;

template<class Label, class T>
class RPTree;

/* Class Definitions */
/*
 * Name             : RPTree
 * Description      : Encapsulates the RPTreeNodes into tree.
 * Data Field(s)    : root_ - Holds the root node of tree
 *                    st_   - Holds the data set associated with tree
 * Function(s)      : RPTree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    RPTree(size_t, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size and
 *                            data set
 *                    RPTree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization 
 */
template<class Label, class T>
class RPTree : public PCATree<Label, T>
{
private:
    static PCATreeNode<Label, T> * build_tree(size_t c,
            DataSet<Label, T> & st, vector<size_t> domain);

public:
    RPTree(DataSet<Label, T> & st);
    RPTree(size_t min_leaf_size, DataSet<Label, T> & st);
    RPTree(ifstream & in, DataSet<Label, T> & st);
};

template<class Label, class T>
PCATreeNode<Label, T> * RPTree<Label, T>::build_tree(size_t min_leaf_size,
        DataSet<Label, T> & st, vector<size_t> domain)
{
    LOG_INFO("Enter build_tree\n");
    LOG_FINE("with min_leaf_size = %ld and domain.size = %ld\n", min_leaf_size, domain.size());
    if (domain.size() < min_leaf_size) {
        LOG_INFO("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new PCATreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);

    //Find a random vector
    size_t dimension = (*subst[0]).size();
    vector<double> split_dir = random_tie_breaker(dimension);
    
    vector<double> values;
    for (size_t i = 0; i < subst.size(); i++) {
        double product = dot(*subst[i], split_dir);
        values.push_back(product);
    }
    
    double pivot = selector(values, (size_t)(values.size() * 0.5));
    vector<size_t> subdomain_l;
    size_t subdomain_l_lim = (size_t)(values.size() * 0.5);
    LOG_FINE("> l_lim = %ld\n", subdomain_l_lim);
    vector<size_t> subdomain_r; 
    vector<size_t> pivot_pool;
    for (size_t i = 0; i < domain.size(); i++) {
        if (pivot == values[i])
            pivot_pool.push_back(domain[i]);
        else if (pivot > values[i])
            subdomain_l.push_back(domain[i]);
        else
            subdomain_r.push_back(domain[i]);
    }

    size_t pivot_count = pivot_pool.size();
    LOG_FINE("pivot pool size %zu\n", pivot_count);
    while (subdomain_l_lim > subdomain_l.size()) {
        size_t curr = pivot_pool.back();
        pivot_pool.pop_back();
        subdomain_l.push_back(curr);
    }
    while (!pivot_pool.empty()) {
        size_t curr = pivot_pool.back();
        pivot_pool.pop_back();
        subdomain_r.push_back(curr);
    }
    PCATreeNode<Label, T> * result = new PCATreeNode<Label, T>(split_dir, 
            pivot, domain);
    result->set_left(build_tree(min_leaf_size, st, subdomain_l));
    result->set_right(build_tree(min_leaf_size, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_INFO("Exit build_tree\n");
    return result;
}

template<class Label, class T>
RPTree<Label, T>::RPTree(DataSet<Label, T> & st) :
    PCATree<Label, T>(st)
{
    LOG_INFO("RPTree Constructed\n");
    LOG_FINE("with default constructor\n");
}

template<class Label, class T>
RPTree<Label, T>::RPTree(size_t min_leaf_size, DataSet<Label, T> & st) :
    PCATree<Label, T>(st)
{
    LOG_INFO("RPTree Constructed\n");
    LOG_FINE("with min_leaf_size = %ld", min_leaf_size);
    this->set_root(build_tree(min_leaf_size, st, st.get_domain()));
}

template<class Label, class T>
RPTree<Label, T>::RPTree(ifstream & in, DataSet<Label, T> & st) :
    PCATree<Label, T>(in, st)
{
    LOG_INFO("RPTree Constructed\n");
    LOG_FINE("with input stream\n");
}

#endif
