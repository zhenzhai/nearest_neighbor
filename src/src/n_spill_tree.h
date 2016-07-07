/* 
 * File             : n_spill_tree.h
 * Summary          : Infrastructure to hold a tree with n+1 children.
 *                    Children are splited using n pivots
 *                    along with spill in each child node.
 */
#ifndef NSPILL_TREE_H_
#define NSPILL_TREE_H_

#include <queue>
#include <map>
#include <array>
#include "logging.h"
#include "data_set.h"
using namespace std;

/* Class Prototypes */

template<class Label, class T>
class NSpillTreeNode;

template<class Label, class T>
class NSpillTree;

/* Class Definitions */

/* 
 * Name             : NSpillTreeNode
 * Description      : Data structure to hold a node of a NSpillTree
 * Data Field(s)    : index_            - The max variance index
 *                    splits_           - The number of splits 'n'
 *                    dimension_        - The dimension of feature, used in de-serialization
 *                    left_pivots_      - vector of n left pivots
 *                    right_pivots_     - vector of n right pivots
 *                    tie_breaker_      - tie breaker vector
 *                    tie_left_pivots_  - vector of n left pivots in tie breaker
 *                    tie_right_pivots_ - vector of n right pivots in tie breaker
 *                    children_ - vector of n+1 subtree nodes
 *                    domain_   - vectors out of vector space in data set
 * Functions(s)     : NSpillTreeNode(const vector<size_t>)
 *                              - Create a NSpillTreeNode of given domain (leaf)
 *                    NSpillTreeNode(size_t, T, size_t, vector<size_t>)
 *                              - Create a NSpillTreeNode of given domain (non-leaf)
 *                    NSpillTreeNode(ifstream &)
 *                              - Creates a NSpillTreeNode through de-serialization
 *                    size_t get_index() const
 *                              - Gets index of max variance
 *                    vector<size_t> get_domain() const
 *                              - Returns the domain the node stores
 *                    void save(ofstream &)
 *                              - Serializes node 
 */
template<class Label, class T>
class NSpillTreeNode
{
protected:
    size_t index_;
    size_t splits_;
    size_t dimension_;
    vector<T> pivots_;
    vector<double> tie_breaker_;
    vector<double> tie_left_pivots_;
    vector<NSpillTreeNode *> children_;
    vector<size_t> domain_;
public:
    NSpillTreeNode(const vector<size_t> domain);
    NSpillTreeNode(size_t max_var_index, size_t splits, size_t dimension,
                   vector<T> pivots, vector<double> tie_breaker, vector<double> tie_left_pivots,
                   vector<NSpillTreeNode *> children, vector<size_t> domain);
    NSpillTreeNode(ifstream & in);
    ~NSpillTreeNode();
    size_t get_index() const
    { return index_; }
    vector<size_t> get_domain() const
    { return domain_; }
    virtual void save(ofstream & out) const;
    
    friend class NSpillTree<Label, T>;
};

/*
 * Name             : NSpillTree
 * Description      : Encapsulates the NSpillTreeNodes into tree.
 * Data Field(s)    : root_ - Holds the root node of tree
 *                    st_   - Holds the data set associated with tree
 * Function(s)      : NSpillTree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    NSpillTree(size_t, double, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size
 *                            and the spill factor
 *                    NSpillTree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization 
 *                    ~NSpillTree()
 *                          - Deconstructor
 *                    NSpillTreeNode<Label, T> * get_root() const
 *                          - Returns the root
 *                    DataSet<Label, T> & get_st() const
 *                          - Returns the set associated with the tree
 *                    void save(ofstream &) const
 *                          - Serializes the tree
 *                    vector<size_t> subdomain(vector<T> *, size_t)
 *                          - Queries the tree for a subdomain
 */
template<class Label, class T>
class NSpillTree
{
private:
    static NSpillTreeNode<Label, T> * build_tree(size_t leaf_size,
            DataSet<Label, T> & st, size_t splits, double spill_factor, vector<size_t> domain);
protected:
    NSpillTreeNode<Label, T> * root_;
    DataSet<Label, T> & st_;
    NSpillTree(DataSet<Label, T> & st);
public:
    NSpillTree(size_t leaf_size, size_t splits, double spill_factor, DataSet<Label, T> & st);
    NSpillTree(ifstream & in, size_t splits, DataSet<Label, T> & st);
    ~NSpillTree();
    NSpillTreeNode<Label, T> * get_root() const
    { return root_; }
    DataSet<Label, T> & get_st() const
    { return st_; }
    void set_root(NSpillTreeNode<Label, T> * root)
    { root_ = root; }
    virtual void save(ofstream & out) const;
    virtual vector<size_t> subdomain(vector<T> * query, size_t l_c = 0);
};




/**************** Private Functions *****************/

/*
 * Build tree of given mininum leaf size <min_leaf_size>,
 * dataset <st>, number of splits <splits>,
 * spill factor <spill_factor>, and vector space <domain>
 */
template<class Label, class T>
NSpillTreeNode<Label, T> * NSpillTree<Label, T>::build_tree(size_t leaf_size,
        DataSet<Label, T> & st, size_t splits, double spill_factor, vector<size_t> domain)
{
    LOG_INFO("Enter build_tree\n");
    LOG_FINE("with min_leaf_size = %ld, splits = %ld, and domain size = %ld\n", leaf_size, splits, domain.size());
    if (domain.size() < leaf_size) {
        LOG_INFO("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new NSpillTreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);
    
    //find max variance index
    size_t mx_var_index = max_variance_index(subst);
    
    //get all the values at the max variance index
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++) {
        values.push_back((*subst[i])[mx_var_index]);
    }
    
    //create and calculate the split pivots
    vector<T> left_pivots, right_pivots, pivots;
    size_t child_size = (size_t)(values.size()/splits - values.size()*spill_factor);
    size_t full_child_size = (size_t)(values.size()/splits);
    size_t half_spill_size = (size_t)(values.size() * spill_factor);
    size_t spill_size = half_spill_size * 2;
    LOG_FINE("> lim = %ld\n", full_child_size);
    for (int i=1; i<splits; i++) { // only (#_of_splits - 1) pivots
        pivots.push_back(selector(values, full_child_size * i));
        left_pivots.push_back(selector(values, full_child_size * i - half_spill_size));
        right_pivots.push_back(selector(values, full_child_size * i + half_spill_size));
    }
    
    
    //split the values into groups
    vector<vector<size_t>> left_pivot_pools(splits-1);
    vector<vector<size_t>> right_pivot_pools(splits-1);
    vector<vector<size_t>> children(splits);
    vector<vector<size_t>> spills(splits-1);
    for (int i=0; i<domain.size(); i++) {
        bool pushed = false;
        int j = 0;
        while(!pushed && j<splits-1) {
            if (values[i] < left_pivots[j]) {
                children[j].push_back(domain[i]);
                pushed = true;
            }
            else if (values[i] == left_pivots[j]) {
                left_pivot_pools[j].push_back(domain[i]);
                pushed = true;
            }
            else if (left_pivots[j] < values[i] && values[i] < right_pivots[j]) {
                spills[j].push_back(domain[i]);
                pushed = true;
            }
            else if (values[i] == right_pivots[j]) {
                right_pivot_pools[j].push_back(domain[i]);
                pushed = true;
            }
            j++;
        }
        if (!pushed) {
            children[splits-1].push_back(domain[i]);
        }
    }
    
    
    //distribute pivot pool to children nodes
    //dot a random vector to break tie then do split again
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    
    //store tie breaker pivots in update_pool
    vector<double> tie_left_pivots(splits-1);
    vector<double> tie_right_pivots(splits-1);
    for (int i = 0; i < left_pivot_pools.size(); i++) {
        //extract the vectors from dataset
        DataSet<Label, T> left_pivot_vectors = st.subset(left_pivot_pools[i]);
        
        //update pool using random tie breaker
        vector<double> updated_left_pool;
        for (int j = 0; j < left_pivot_vectors.size(); j++) {
            double product = dot(*left_pivot_vectors[j], tie_breaker);
            updated_left_pool.push_back(product);
        }
        
        //Begin to use left_pool until it is all used up
        size_t left_pool_size = updated_left_pool.size();
        size_t filled_size = 0;
        bool use_right_pool = false;
        double previous_tie_pivot = selector(updated_left_pool, 1);
        double tie_pivot = previous_tie_pivot;
        
        //initial push to left children, because == is not dealt with in the while loop below
        for (int j = 0; j < left_pivot_vectors.size(); j++) {
            if (previous_tie_pivot == updated_left_pool[j]) {
                children[i].push_back(left_pivot_pools[i][j]);
                left_pool_size--;
            }
        }

        while (left_pool_size > 0) {
            //find the tie_pivots and distribute to the left child
            size_t to_fill_left = child_size - children[i].size();
            
            // If left pool can't fill current node
            if (left_pool_size < to_fill_left) {
                tie_pivot = selector(updated_left_pool, updated_left_pool.size());
            }
            else {
                filled_size = updated_left_pool.size()-left_pool_size;
                tie_pivot = selector(updated_left_pool, to_fill_left + filled_size);
                tie_left_pivots.push_back(tie_pivot); // this pivot is the left pivot when search
            }
            
            for (int j = 0; j < left_pivot_vectors.size(); j++) {
                if (previous_tie_pivot < updated_left_pool[j] && updated_left_pool[j] <= tie_pivot) {
                    children[i].push_back(left_pivot_pools[i][j]);
                    left_pool_size--;
                }
            }
            previous_tie_pivot = tie_pivot;
            
            //Check whether left_pool_size is empty
            if (left_pool_size == 0)
                break;
            
            // find the tie_pivots and distribute to the spill node
            size_t to_fill_spill = spill_size - spills[i].size();

            // If left pool can't fill spill node, need to use right pool
            if (left_pool_size < to_fill_spill) {
                use_right_pool = true;
                //Check whether left_pool_size is empty
                if (left_pool_size == 0)
                    break;
                tie_pivot = selector(updated_left_pool, updated_left_pool.size());
            }
            else { //left pool can fill up the spill node, therefore left_pivot == right_pivot.
                    //No need to look at right pivot pool because it will be the same as left pivot pool
                filled_size = updated_left_pool.size()-left_pool_size;
                tie_pivot = selector(updated_left_pool, to_fill_spill + filled_size);
                tie_right_pivots.push_back(tie_pivot); // this pivot is the right pivot when search
            }
            
            for (int j = 0; j < left_pivot_vectors.size(); j++) {
                if (previous_tie_pivot < updated_left_pool[j] && updated_left_pool[j] <= tie_pivot) {
                    spills[i].push_back(left_pivot_pools[i][j]);
                    left_pool_size--;
                }
            }
            previous_tie_pivot = tie_pivot;
            
            // increse i only when left pool is not empty
            if (left_pool_size > 0)
                i++;
        }
        
        if (!use_right_pool)
            continue;
        
        //extract the vectors from dataset
        DataSet<Label, T> right_pivot_vectors = st.subset(right_pivot_pools[i]);
        
        //update pool using random tie breaker
        vector<double> updated_right_pool;
        for (int j = 0; j < right_pivot_vectors.size(); j++) {
            double product = dot(*right_pivot_vectors[j], tie_breaker);
            updated_right_pool.push_back(product);
        }
        
        //Begin to use right_pool until it is all used up
        size_t right_pool_size = updated_right_pool.size();
        previous_tie_pivot = selector(updated_right_pool, 1);
        tie_pivot = previous_tie_pivot;
        
        for (int j = 0; j < right_pivot_vectors.size(); j++) {
            if (previous_tie_pivot == updated_right_pool[j]) {
                spills[i].push_back(right_pivot_pools[i][j]);
                right_pool_size--;
            }
        }
    
        while (right_pool_size > 0) {
            size_t to_fill_spill = spill_size - spills[i].size();
            if (right_pool_size <= to_fill_spill) {
                tie_pivot = selector(updated_right_pool, updated_right_pool.size());
            }
            else {
                filled_size = updated_right_pool.size()-right_pool_size;
                tie_pivot = selector(updated_right_pool, to_fill_spill + filled_size);
                tie_right_pivots.push_back(tie_pivot); // this pivot is the right pivot when search
            }
            
            for (int j = 0; j < right_pivot_vectors.size(); j++) {
                if (previous_tie_pivot < updated_right_pool[j] && updated_right_pool[j] <= tie_pivot) {
                    spills[i].push_back(right_pivot_pools[i][j]);
                    right_pool_size--;
                }
            }
            previous_tie_pivot = tie_pivot;
            
            //Check whether right_pool_size is empty
            if (right_pool_size == 0)
                break;
            
            size_t to_fill_right = child_size - children[i+1].size();
            if (right_pool_size <= to_fill_right) {
                tie_pivot = selector(updated_right_pool, updated_right_pool.size());
            }
            else {
                filled_size = updated_right_pool.size()-right_pool_size;
                tie_pivot = selector(updated_right_pool, to_fill_right + filled_size);
                tie_left_pivots.push_back(tie_pivot); // this pivot is the left pivot when search
            }
            
            for (int j = 0; j < right_pivot_vectors.size(); j++) {
                if (previous_tie_pivot < updated_right_pool[j] && updated_right_pool[j] <= tie_pivot) {
                    children[i+1].push_back(right_pivot_pools[i][j]);
                    right_pool_size--;
                }
            }
            
            // increse i only when right pool is not empty
            if (right_pool_size > 0)
                i++;
        }
    }
    
    //distribute spill
    for (int i=0; i<spills.size(); i++) {
        for (int j=0; j<spills[i].size(); j++) {
            children[i].push_back(spills[i][j]);
            children[i+1].push_back(spills[i][j]);
        }
    }

    
    //call build tree recursively to build tree
    vector<NSpillTreeNode<Label, T> *> children_vector;
    for (int i=0; i<splits; i++) {
        NSpillTreeNode<Label, T> * node = build_tree(leaf_size, st, splits, spill_factor, children[i]);
        children_vector.push_back(node);
    }
    //return newly built tree
    NSpillTreeNode<Label, T> * result = new NSpillTreeNode<Label, T>(mx_var_index, splits, dimension, pivots, tie_breaker, tie_left_pivots, children_vector, domain);
    LOG_INFO("Exit build_tree\n");
    return result;
}

/* Public Functions */

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(const vector<size_t> domain) :
  index_ (0),
  splits_(0),
  dimension_ (0),
  pivots_ (NULL),
  tie_breaker_(NULL),
  tie_left_pivots_(NULL),
  children_ (NULL),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(
		size_t index,
        size_t splits,
        size_t dimension,
        vector<T> pivots,
        vector<double> tie_breaker,
        vector<double> tie_left_pivots,
		vector<NSpillTreeNode *> children,
		vector<size_t> domain) :
  index_ (index),
  splits_ (splits),
  dimension_ (dimension),
  pivots_ (pivots),
  tie_breaker_ (tie_breaker),
  tie_left_pivots_ (tie_left_pivots),
  children_ (children),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with index = %ld, domain.size = %ld, children = %ld\n", index,
            domain.size(), splits);
}

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(ifstream & in)
{
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with input stream\n");
    in.read((char *)&index_, sizeof(size_t));
    in.read((char *)&splits_, sizeof(size_t));
    size_t number_of_pivots = splits_ - 1;
    if (splits_ == 0) //make sure it doesn't become negative
        number_of_pivots = 0;
    size_t pivots = number_of_pivots;
    while (pivots--)
    {
        T pivot;
        in.read((char *)&pivot, sizeof(T));
        pivots_.push_back(pivot);
    }
    size_t sz;
    in.read((char *)&sz, sizeof(size_t));
    LOG_FINE("> sz = %ld\n", sz);
    while (sz--)
    {
        size_t v;
        in.read((char *)&v, sizeof(size_t));
        domain_.push_back(v);
    }
    in.read((char *)&dimension_, sizeof(size_t));
    size_t dimen = dimension_;
    LOG_FINE("> dimension = %ld\n", dimen);
    while (dimen--)
    {
        size_t v;
        in.read((char *)&v, sizeof(double));
        tie_breaker_.push_back(v);
    }
    pivots = number_of_pivots;
    while (pivots--)
    {
        double tie_pivot;
        in.read((char *)&tie_pivot, sizeof(double));
        tie_left_pivots_.push_back(tie_pivot);
    }
}

template<class Label, class T>
void NSpillTreeNode<Label, T>::save(ofstream & out) const
{
    LOG_INFO("Saving NSpillTreeNode\n");
    LOG_FINE("> domain.size = %ld\n", domain_.size());
    out.write((char *)&index_, sizeof(size_t));
    out.write((char *)&splits_, sizeof(size_t));
    out.write((char *)&pivots_[0],
              sizeof(T) * pivots_.size());
        size_t sz = domain_.size();
    out.write((char *)&sz, sizeof(size_t)); 
    out.write((char *)&domain_[0], sizeof(size_t) * sz);
    out.write((char *)&dimension_, sizeof(size_t));
    out.write((char *)&tie_breaker_[0], sizeof(double) * dimension_);
    out.write((char *)&tie_left_pivots_[0],
              sizeof(double) * tie_left_pivots_.size());
}

template<class Label, class T>
NSpillTreeNode<Label, T>::~NSpillTreeNode()
{
    LOG_FINE("Deleted children\n");
    for (int i=0; i<children_.size(); i++) {
        delete children_[i];
    }
    LOG_INFO("NSpillTreeNode Deconstructed\n");
}

template<class Label, class T>
NSpillTree<Label, T>::NSpillTree(DataSet<Label, T> & st) :
  root_ (NULL),
  st_ (st)
{ 
    LOG_INFO("NSpillTree Constructed\n");
    LOG_FINE("with default constructor\n");
}

template<class Label, class T>
NSpillTree<Label, T>::NSpillTree(size_t leaf_size, size_t splits, double spill_factor, DataSet<Label, T> & st) :
  root_ (build_tree(leaf_size, st, splits, spill_factor, st.get_domain())),
  st_ (st)
{ 
    LOG_INFO("NSpillTree Constructed\n");
    LOG_FINE("with leaf_size = %ld, spill_factor = %lf\n", leaf_size, spill_factor);
}

template<class Label, class T>
NSpillTree<Label, T>::~NSpillTree()
{
    if (root_) delete root_;
    LOG_INFO("NSpillTree Deconstructed\n");
}

template<class Label, class T>
NSpillTree<Label, T>::NSpillTree(ifstream & in, size_t splits, DataSet<Label, T> & st) :
  st_ (st)
{
    LOG_INFO("NSpillTree Constructed\n");
    LOG_FINE("with input stream\n");
    queue<NSpillTreeNode<Label, T> **> to_load;
    to_load.push(&root_);
    while (!to_load.empty())
    {
        NSpillTreeNode<Label, T> ** cur = to_load.front();
        to_load.pop();
        bool exist;
        in.read((char *)&exist, sizeof(bool));
        if (!exist) {
            *cur = NULL; 
            continue;
        }
        *cur = new NSpillTreeNode<Label, T>(in);
        
        for (int i=0; i<splits; i++){
            (*cur)->children_.push_back(NULL);
        }
        for (int i=0; i<splits; i++){
            to_load.push(&((*cur)->children_[i]));
        }
    }
}


template<class Label, class T>
void NSpillTree<Label, T>::save(ofstream & out) const
{
    LOG_INFO("Saving NSpillTreeNode\n");
    queue<NSpillTreeNode<Label, T> *> to_save;
    to_save.push(root_);
    while (!to_save.empty()) {
        NSpillTreeNode<Label, T> * cur = to_save.front();
        to_save.pop();
        bool exists = cur != NULL;
        out.write((char *)&exists, sizeof(bool)); 
        if (exists) {
            cur->save(out);
            if (!cur->children_.empty()) {
                for (int i=0; i < cur->splits_; i++) {
                    to_save.push(cur->children_[i]);
                }
            } else {
                to_save.push(NULL);
            }
        }
    }
}

template<class Label, class T>
vector<size_t> NSpillTree<Label, T>::subdomain(vector<T> * query, size_t l_c)
{
    LOG_INFO("Enter subdomain\n");
    LOG_FINE("with lc = %ld\n", l_c);
    queue<NSpillTreeNode<Label, T> *> expl;
    expl.push(root_);
    while (!expl.empty())
    {
        NSpillTreeNode<Label, T> * cur = expl.front();
        expl.pop();
        if (!cur->children_.empty() &&
            cur->domain_.size() >= l_c) {
            size_t splits = cur->splits_;
            bool pushed = false;
            for (int i = 0; i < splits-1; i++) {
                if ((*query)[cur->index_] < cur->pivots_[i]) {
                    expl.push(cur->children_[i]);
                    pushed = true;
                    break;
                }
                else if ((*query)[cur->index_] == cur->pivots_[i]) {
                    vector<double> tie_breaker = cur->tie_breaker_;
                    double product = dot(*query, tie_breaker);
                    //Can use tie_left_pivot when search
                    //It doesn't matters where it goes when the tie range is smaller than the spill range, apply left tie won't hurt.
                    //If tie range is larger than the spill range, left tie alone can determine which leaf to go.
                    if (product <= cur->tie_left_pivots_[i]) {
                        expl.push(cur->children_[i]);
                        pushed = true;
                        break;
                    }
                }
            }
            if (!pushed) {
                expl.push(cur->children_[splits-1]);
            }
        }
        else
            return cur->domain_;
    }
    LOG_INFO("Exit subdomain\n");
    return vector<size_t>();
}

#endif
