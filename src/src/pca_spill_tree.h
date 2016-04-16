/* 
 * File             : pca_spill_tree.h
 * Summary          : Infrastructure to hold a binary space partition spill 
 *                    tree.
 */
#ifndef PCA_SPILL_TREE_H_
#define PCA_SPILL_TREE_H_

#include "pca_tree.h"

#define MIN(x, y) (x) < (y) ? (x) : (y)

/* Class Prototypes */

template<class Label, class T>
class PCASpillTree;

/* Class Definitions */

/* 
 * Name             : PCASpillTree
 * Description      : Encapsulates the PCATreeNodes into a spill tree.
 *                    Effectively acts as identically to PCATree with spillage in
 *                    terms of its nodes.
 * Data Field(s)    : None
 * Functions(s)     : PCASpillTree(size_t, double, DataSet<Label, T> &)
 *                          - Creates a spill tree with given min leaf size
 *                            and the spill factor
 *                    PCASpillTree(ifStream & in, DataSet<Label, T> & st)
 *                          - De-serializes a spill tree
 */
template<class Label, class T>
class PCASpillTree : public PCATree<Label, T>
{
private:
    static PCATreeNode<Label, T> * build_tree(size_t min_leaf_size, double a_value,
            DataSet<Label, T> & st, vector<size_t> domain);
public:
    PCASpillTree(DataSet<Label, T> & st);
    PCASpillTree(size_t c, double a, DataSet<Label, T> & st);
    PCASpillTree(ifstream & in, DataSet<Label, T> & st);
};

template<class Label, class T>
PCATreeNode<Label, T> * PCASpillTree<Label, T>::build_tree(size_t min_leaf_size, double a_value,
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
    vector<double> mx_var_dir = max_eigen_vector(subst, 1000);
    vector<double> values;
    for (size_t i = 0; i < subst.size(); i++)
        values.push_back(dot(*subst[i], mx_var_dir));
    double pivot = selector(values, (size_t)(values.size() * 0.5)); //may not need pivot pool
    double pivot_l = selector(values, (size_t)(values.size() * (0.5 - a_value)));
    double pivot_r = selector(values, (size_t)(values.size() * (0.5 + a_value)));
    size_t subdomain_l_lim = (size_t)(values.size() * (0.5 + a_value));
    size_t subdomain_r_lim = subdomain_l_lim; //(size_t)(values.size() * (1 + 2 * a_value) - subdomain_l_lim);
    LOG_FINE("> l_lim = %ld\n", subdomain_l_lim);
    LOG_FINE("> r_lim = %ld\n", subdomain_r_lim);
    vector<size_t> subdomain_l;
    vector<size_t> subdomain_r;
    vector<size_t> pivot_pool;
    vector<size_t> pivot_l_pool;
    vector<size_t> pivot_r_pool;
    for (size_t i = 0; i < domain.size(); i++) {
        if (pivot == values[i]) {
            pivot_pool.push_back(domain[i]);
        } else if (pivot_l == values[i]) {
            pivot_l_pool.push_back(domain[i]);
        } else if (pivot_r == values[i]) {
            pivot_r_pool.push_back(domain[i]);
        } else {
            if (pivot_l < values[i] && values[i] < pivot_r) {
                subdomain_l.push_back(domain[i]);
                subdomain_r.push_back(domain[i]);
            } else if (values[i] < pivot) {
                subdomain_l.push_back(domain[i]);
            } else {
                subdomain_r.push_back(domain[i]);
            }
        }
    }
    
    //Distribute values in pools
    //TODO: tie breaking
    size_t d_l = MIN(subdomain_l_lim - subdomain_l.size(), pivot_pool.size() + pivot_l_pool.size() + pivot_r_pool.size());
    size_t d_r = MIN(subdomain_r_lim - subdomain_r.size(), pivot_pool.size() + pivot_l_pool.size() + pivot_r_pool.size());
    size_t spill = d_l + d_r - (pivot_pool.size() + pivot_l_pool.size() + pivot_r_pool.size());
    LOG_FINE("> dl = %ld\n", d_l);
    LOG_FINE("> dr = %ld\n", d_r);
    LOG_FINE("> spill = %ld\n", spill);

    for (size_t i = 0; i < d_l - spill; i++) {
        size_t curr;
        if (!pivot_l_pool.empty()) {
             curr = pivot_l_pool.back();
             pivot_l_pool.pop_back();
        } else if (!pivot_pool.empty()) {
             curr = pivot_pool.back();
             pivot_pool.pop_back();
        } else {
             curr = pivot_r_pool.back();
             pivot_r_pool.pop_back();
        }
        subdomain_l.push_back(curr);
    }
    for (size_t i = 0; i < d_r - spill; i++) {
        size_t curr;
        if (!pivot_r_pool.empty()) {
             curr = pivot_r_pool.back();
             pivot_r_pool.pop_back();
        } else if (!pivot_pool.empty()) {
             curr = pivot_pool.back();
             pivot_pool.pop_back();
        } else {
             curr = pivot_l_pool.back();
             pivot_l_pool.pop_back();
        }
        subdomain_r.push_back(curr);
    }
    for (size_t i = 0; i < spill; i++) {
        size_t curr;
        if (!pivot_pool.empty()) {
             curr = pivot_pool.back();
             pivot_pool.pop_back();
        } else if (!pivot_l_pool.empty()) {
             curr = pivot_l_pool.back();
             pivot_l_pool.pop_back();
        } else {
             curr = pivot_r_pool.back();
             pivot_r_pool.pop_back();
        }
        subdomain_l.push_back(curr);
        subdomain_r.push_back(curr);
    }
    PCATreeNode<Label, T> * result = new PCATreeNode<Label, T>
            (mx_var_dir, pivot, domain);
    result->set_left(build_tree(min_leaf_size, a_value, st, subdomain_l));
    result->set_right(build_tree(min_leaf_size, a_value, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_INFO("Exit build_tree\n");
    return result;
}

/* Private Functions */

template<class Label, class T>
PCASpillTree<Label, T>::PCASpillTree(DataSet<Label, T> & st) :
  PCATree<Label, T>(st)
{ 
    LOG_INFO("PCASpillTree Constructed\n"); 
    LOG_FINE("with default constructor\n");
}

/* Public Functions */

template<class Label, class T>
PCASpillTree<Label, T>::PCASpillTree(size_t c, double a, DataSet<Label, T> & st) :
  PCATree<Label, T>(st)
{ 
    LOG_INFO("PCASpillTree Constructed\n"); 
    LOG_FINE("with c = %ld, a = %lf\n", c, a);
    this->set_root(build_tree(c, a, st, st.get_domain())); 
}

template<class Label, class T>
PCASpillTree<Label, T>::PCASpillTree(ifstream & in, DataSet<Label, T> & st) :
  PCATree<Label, T>(in, st)
{ 
    LOG_INFO("PCASpillTree Constructed\n"); 
    LOG_FINE("with input stream\n");
}

#endif
