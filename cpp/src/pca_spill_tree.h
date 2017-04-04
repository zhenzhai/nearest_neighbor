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
    static PCATreeNode<Label, T> * build_tree(size_t min_leaf_size, double spill_factor,
            DataSet<Label, T> & st, vector<size_t> domain);
public:
    PCASpillTree(DataSet<Label, T> & st);
    PCASpillTree(size_t min_leaf_size, double a, DataSet<Label, T> & st);
    PCASpillTree(ifstream & in, DataSet<Label, T> & st);
};

template<class Label, class T>
PCATreeNode<Label, T> * PCASpillTree<Label, T>::build_tree(size_t min_leaf_size, double spill_factor,
        DataSet<Label, T> & st, vector<size_t> domain)
{
    LOG_FINE("Enter build_tree\n");
    LOG_FINE("with min_leaf_size = %ld and domain.size = %ld\n", min_leaf_size, domain.size());
    if (domain.size() < min_leaf_size) {
		LOG_FINE("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new PCATreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);

	/*find dominant eigenvector*/
    vector<double> mx_var_dir = max_eigen_vector(subst, 1000);

	/*project all the data at the dominant eigenvector*/
    vector<double> values;
    for (size_t i = 0; i < subst.size(); i++)
        values.push_back(dot(*subst[i], mx_var_dir));

	/*find size limit for spill, left/right child, and half child*/
	size_t spill_size_lim = (size_t)(values.size() * spill_factor * 2);
	size_t child_size_lim = (size_t)(values.size() * (0.5 - spill_factor));
	size_t half_size_lim = (size_t)(values.size() * 0.5);

	double pivot = selector(values, half_size_lim);
	double pivot_l = selector(values, child_size_lim);
	double pivot_r = selector(values, child_size_lim + spill_size_lim);

	size_t subdomain_l_lim = child_size_lim + spill_size_lim;
	size_t subdomain_r_lim = subdomain_l_lim;
	LOG_FINE("> l_lim = %ld\n", subdomain_l_lim);
	LOG_FINE("> r_lim = %ld\n", subdomain_r_lim);

	/*Distribute to left and right child*/
	vector<size_t> subdomain_l;
	vector<size_t> subdomain_r;
	for (size_t i = 0; i < domain.size(); i++) {
		if (values[i] <= pivot_l) {
			subdomain_l.push_back(domain[i]);
		}
		else if (pivot_l < values[i] && values[i] <= pivot_r) {
			subdomain_l.push_back(domain[i]);
			subdomain_r.push_back(domain[i]);
		}
		else {
			subdomain_r.push_back(domain[i]);
		}
	}

	PCATreeNode<Label, T> * result = new PCATreeNode<Label, T> (mx_var_dir, pivot, domain);
    result->set_left(build_tree(min_leaf_size, spill_factor, st, subdomain_l));
    result->set_right(build_tree(min_leaf_size, spill_factor, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_FINE("Exit build_tree\n");
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
PCASpillTree<Label, T>::PCASpillTree(size_t min_leaf_size, double spill_factor, DataSet<Label, T> & st) :
  PCATree<Label, T>(st)
{ 
    LOG_INFO("PCASpillTree Constructed\n"); 
    LOG_FINE("with min_leaf_size = %ld, spill_factor = %lf\n", min_leaf_size, spill_factor);
    this->set_root(build_tree(min_leaf_size, spill_factor, st, st.get_domain()));
}

template<class Label, class T>
PCASpillTree<Label, T>::PCASpillTree(ifstream & in, DataSet<Label, T> & st) :
  PCATree<Label, T>(in, st)
{ 
    LOG_INFO("PCASpillTree Constructed\n"); 
    LOG_FINE("with input stream\n");
}

#endif
