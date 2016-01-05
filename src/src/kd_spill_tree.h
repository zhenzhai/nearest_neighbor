/* 
 * File             : kd_spill_tree.h
 * Date             : 2014-5-29
 * Summary          : Infrastructure to hold a kd spill tree.
 */
#ifndef KD_SPILL_TREE_H_
#define KD_SPILL_TREE_H_

#include "kd_tree.h"

#define MIN(x, y) (x) < (y) ? (x) : (y)

/* Class Prototypes */

template<class Label, class T>
class KDSpillTree;

/* Class Definitions */

/* 
 * Name             : KDSpillTree
 * Description      : Encapsulates the KDTreeNodes into a spill tree.
 *                    Effectively acts as identically to KDTree with spillage in
 *                    terms of its nodes.
 * Data Field(s)    : None
 * Functions(s)     : KDSpillTree(size_t, double, DataSet<Label, T> &)
 *                          - Creates a spill tree with given min leaf size
 *                            and the spill factor
 *                    KDSpillTree(ifStream & in, DataSet<Label, T> & st)
 *                          - De-serializes a spill tree
 */
template<class Label, class T>
class KDSpillTree : public KDTree<Label, T>
{
private:
    static KDTreeNode<Label, T> * build_tree(size_t c, double a,
            DataSet<Label, T> & st, vector<size_t> domain);
    KDSpillTree(DataSet<Label, T> & st);
public:
    KDSpillTree(size_t c, double a, DataSet<Label, T> & st);
    KDSpillTree(ifstream & in, DataSet<Label, T> & st);
};

/* Private Functions */

template<class Label, class T>
KDTreeNode<Label, T> * KDSpillTree<Label, T>::build_tree(size_t c, double a,
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
    vector<double> vars = variances(subst);
    size_t mx_var_index = max_variance_index(subst, vars);
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++)
        values.push_back((*subst[i])[mx_var_index]);
    double pivot = selector(values, (size_t)(values.size() * 0.5));
    double pivot_l = selector(values, (size_t)(values.size() * (0.5 - a)));
    double pivot_r = selector(values, (size_t)(values.size() * (0.5 + a)));
    vector<size_t> subdomain_l;
    size_t subdomain_l_lim = (size_t)(values.size() * (0.5 + a));
    size_t subdomain_r_lim = (size_t)(values.size() * (1 + 2*a) - subdomain_l_lim);
    LOG_FINE("> l_lim = %ld\n", subdomain_l_lim);
    LOG_FINE("> r_lim = %ld\n", subdomain_l_lim);
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
            } else {
                if (values[i] < pivot) {
                    subdomain_l.push_back(domain[i]);
                } else {
                    subdomain_r.push_back(domain[i]);
                }
            }
        }
    }
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
    KDTreeNode<Label, T> * result = new KDTreeNode<Label, T>
            (mx_var_index, pivot, domain);
    result->set_left(build_tree(c, a, st, subdomain_l));
    result->set_right(build_tree(c, a, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_INFO("Exit build_tree\n");
    return result;
}

template<class Label, class T>
KDSpillTree<Label, T>::KDSpillTree(DataSet<Label, T> & st) :
  KDTree<Label, T>(st)
{ 
    LOG_INFO("KDSpillTree Constructed\n"); 
    LOG_FINE("with default constructor\n");
}

/* Public Functions */

template<class Label, class T>
KDSpillTree<Label, T>::KDSpillTree(size_t c, double a, DataSet<Label, T> & st) :
  KDTree<Label, T>(st)
{ 
    LOG_INFO("KDSpillTree Constructed\n"); 
    LOG_FINE("with c = %ld, a = %lf\n", c, a);
    this->set_root(build_tree(c, a, st, st.get_domain())); 
}

template<class Label, class T>
KDSpillTree<Label, T>::KDSpillTree(ifstream & in, DataSet<Label, T> & st) :
  KDTree<Label, T>(in, st)
{ 
    LOG_INFO("KDSpillTree Constructed\n"); 
    LOG_FINE("with input stream\n");
}

#endif
