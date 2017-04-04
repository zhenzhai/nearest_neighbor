/* 
 * File             : kd_spill_tree.h
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
    static KDTreeNode<Label, T> * build_tree(size_t min_leaf_size, double spill_factor,
            DataSet<Label, T> & st, vector<size_t> domain);
public:
	KDSpillTree(DataSet<Label, T> & st);
    KDSpillTree(size_t min_leaf_size, double a_value, DataSet<Label, T> & st);
    KDSpillTree(ifstream & in, DataSet<Label, T> & st);
};

/* Private Functions */

template<class Label, class T>
KDTreeNode<Label, T> * KDSpillTree<Label, T>::build_tree(size_t min_leaf_size, double spill_factor,
        DataSet<Label, T> & st, vector<size_t> domain)
{
    LOG_INFO("Enter build_tree\n");
    LOG_FINE("with min_leaf_size = %ld and domain.size = %ld\n", min_leaf_size, domain.size());
    if (domain.size() < min_leaf_size) {
        LOG_INFO("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new KDTreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);
    
    /*find max variance index*/
    size_t mx_var_index = max_variance_index(subst);
    
    /*get all the values at the max variance index*/
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++)
        values.push_back((*subst[i])[mx_var_index]);
    
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
    
	/*Distribute to left, right, and pivot node*/
    vector<size_t> subdomain_l;
    vector<size_t> subdomain_r;
    vector<size_t> pivot_l_pool;
    vector<size_t> pivot_r_pool;
    vector<size_t> pivot_pool;
    vector<size_t> simple_l;
    vector<size_t> simple_r;
    vector<size_t> spill;
    for (size_t i = 0; i < domain.size(); i++) {
        if (values[i] < pivot_l) {
            subdomain_l.push_back(domain[i]);
        } else if (values[i] == pivot_l) {
            pivot_l_pool.push_back(domain[i]);
        } else if (pivot_l < values[i] && values[i] < pivot_r) {
            spill.push_back(domain[i]);
        } else if (values[i] == pivot_r) {
            pivot_r_pool.push_back(domain[i]);
        } else {
            subdomain_r.push_back(domain[i]);
        }
        if (values[i] < pivot) {
            simple_l.push_back(domain[i]);
        } else if (values[i] == pivot) {
            pivot_pool.push_back(domain[i]);
        } else {
            simple_r.push_back(domain[i]);
        }
    }
    
    /*Distribute values in pools by doing tie breaking
    dot a random vector then do split again*/
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    
    /*extract the vectors from dataset*/
    DataSet<Label, T> left_pool_vectors = st.subset(pivot_l_pool);
    DataSet<Label, T> right_pool_vectors = st.subset(pivot_r_pool);
    DataSet<Label, T> pivot_pool_vectors = st.subset(pivot_pool);
    
    /*update left pool using random tie breaker*/
    vector<double> update_left_pool;
    for (int j = 0; j < left_pool_vectors.size(); j++) {
        double product = dot(*left_pool_vectors[j], tie_breaker);
        update_left_pool.push_back(product);
    }

	/*update right pool using random tie breaker*/
    vector<double> update_right_pool;
    for (int j = 0; j < right_pool_vectors.size(); j++) {
        double product = dot(*right_pool_vectors[j], tie_breaker);
        update_right_pool.push_back(product);
    }
    
	/*update pivot pool using random tie breaker*/
    vector<double> update_pivot_pool;
    for (int j = 0; j < pivot_pool_vectors.size(); j++) {
        double product = dot(*pivot_pool_vectors[j], tie_breaker);
        update_pivot_pool.push_back(product);
    }
    
    /*tie pivot used in search*/
    double tie_pivot = selector(update_pivot_pool, half_size_lim - simple_l.size());
    
	/*Distribute left tie pivot pool to left child and spill*/
    double tie_pivot_l = selector(update_left_pool, child_size_lim - subdomain_l.size());
    size_t current_l_pool_size = update_left_pool.size();
    for (int i = 0; i < update_left_pool.size(); i++) {
        if (update_left_pool[i] <= tie_pivot_l) {
            subdomain_l.push_back(pivot_l_pool[i]);
            current_l_pool_size--;
        }
    }
    
	/*Distribute left tie pivot pool to spill*/
    size_t to_fill_spill = spill_size_lim - spill.size();
    size_t filled_left = update_left_pool.size() - current_l_pool_size;
    double tie_pivot_r;
    if (to_fill_spill > current_l_pool_size) {
        for (int i = 0; i < update_left_pool.size(); i++) {
            if (update_left_pool[i] > tie_pivot_l) {
                spill.push_back(pivot_l_pool[i]);
                current_l_pool_size--;
            }
        }
		
		/*current_l_pool_size sould be 0 now*/
		if (current_l_pool_size != 0) {
			LOG_ERROR("KD Spill Trees: Current left pool should be zero after filled up spill!(1)\n");
		}
        
        to_fill_spill = spill_size_lim - spill.size();
        if (to_fill_spill > 0) {/*left pivot pool is empty but spill is not full, take from right pivot pool*/
            tie_pivot_r = selector(update_right_pool, to_fill_spill);
            for (int i = 0; i < update_right_pool.size(); i++) {
                if (update_right_pool[i] <= tie_pivot_r) {
                    spill.push_back(pivot_r_pool[i]);
                } else {
                    subdomain_r.push_back(pivot_r_pool[i]);
                }
            }
        }
    }
	else if (to_fill_spill < current_l_pool_size)
		/*In this case, left_pivot == right_pivot
		no need to look at right_pivot pool
		because right_pivot pool will be exactly the same as left pivot pool.*/
	{
		if (pivot_l != pivot_r) {
            LOG_ERROR("KD Spill Trees: Left pivot should be equal to right pivot!\n");
        }
		/*Select the right pivot tie breaker from the left pivot pool*/
        tie_pivot_r = selector(update_left_pool, to_fill_spill + filled_left);
        for (int i = 0; i < update_left_pool.size(); i++) {
            if (tie_pivot_l < update_left_pool[i] && update_left_pool[i] <= tie_pivot_r) {
                spill.push_back(pivot_l_pool[i]);
                current_l_pool_size--;
            } else if(update_left_pool[i] > tie_pivot_r) {
                subdomain_r.push_back(pivot_l_pool[i]);
            }
        }
    }
	else /*spill_size_lim == current_l_pool_size
		 put left pivot pool in spill
		 and put right pivot pool to right child*/
	{
		for (int i = 0; i < update_left_pool.size(); i++) {
			if (update_left_pool[i] > tie_pivot_l) {
				spill.push_back(pivot_l_pool[i]);
				current_l_pool_size--;
			}
		}

		/*current_l_pool_size sould be 0 now*/
		if (current_l_pool_size != 0) {
			LOG_ERROR("KD Spill Trees: Current left pool should be zero after filled up spill!(2)\n");
		}
	}
    
    /*Distribute spill*/
    for (int i=0; i<spill.size(); i++) {
            subdomain_l.push_back(spill[i]);
            subdomain_r.push_back(spill[i]);
    }

    KDTreeNode<Label, T> * result = new KDTreeNode<Label, T>
            (mx_var_index, pivot, domain, dimension, tie_pivot, tie_breaker);
    result->set_left(build_tree(min_leaf_size, spill_factor, st, subdomain_l));
    result->set_right(build_tree(min_leaf_size, spill_factor, st, subdomain_r));
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_FINE("Exit build_tree\n");
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
KDSpillTree<Label, T>::KDSpillTree(size_t min_leaf_size, double spill_factor, DataSet<Label, T> & st) :
  KDTree<Label, T>(st)
{ 
    LOG_INFO("KDSpillTree Constructed\n"); 
    LOG_FINE("with min_leaf_size = %ld, spill_factor = %lf\n", min_leaf_size, spill_factor);
    this->set_root(build_tree(min_leaf_size, spill_factor, st, st.get_domain()));
}

template<class Label, class T>
KDSpillTree<Label, T>::KDSpillTree(ifstream & in, DataSet<Label, T> & st) :
  KDTree<Label, T>(in, st)
{ 
    LOG_INFO("KDSpillTree Constructed\n"); 
    LOG_FINE("with input stream\n");
}

#endif
