/* 
 * File             : n_splits_tree.h
 * Date             : 2015-5-29
 * Summary          : Infrastructure to hold a n splits tree.
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
 * Data Field(s)    : index_    - The max variance index
 *                    pivots_   - vectors of pivots
 *                    children_ - vector of subtree nodes
 *                    domain_   - vectors out of vector space in data set
 * Functions(s)     : NSpillTreeNode(const vector<size_t>)
 *                              - Create a NSpillTreeNode of given domain (leaf)
 *                    NSpillTreeNode(size_t, T, size_t, vector<size_t>)
 *                              - Create a NSpillTreeNode of given domain (non-leaf)
 *                    NSpillTreeNode(ifstream &)
 *                              - Creates a NSpillTreeNode through de-serialization
 *                    size_t get_index() const
 *                              - Gets index of max variance
 *                    NSpillTreeNode * get_children() const
 *                              - Returns pointer to array of subtree nodes
 *                    vector<size_t> get_domain() const
 *                              - Returns the domain the node stores
 *                    void set_children(NSpillTreeNode *)
 *                              - Sets the subtree nodes
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
    vector<T> left_pivots_;
    vector<T> right_pivots_;
    vector<double> tie_breaker_;
    vector<double> tie_left_pivots_;
    vector<double> tie_right_pivots_;
    vector<NSpillTreeNode *> children_;
    vector<size_t> domain_;
public:
    NSpillTreeNode(const vector<size_t> domain);
    NSpillTreeNode(size_t dimension, size_t max_var_index, vector<T> left_pivots,
                   vector<T> right_pivots, size_t splits, vector<double> tie_breaker,
                   vector<double> tie_left_pivots, vector<double> tie_right_pivots,
                   vector<NSpillTreeNode *> children, vector<size_t> domain);
    NSpillTreeNode(ifstream & in);
    ~NSpillTreeNode();
    size_t get_dimension() const
    { return dimension_; }
    size_t get_index() const
    { return index_; }
    vector<T> get_left_pivots() const
    { return left_pivots_; }
    vector<T> get_right_pivots() const
    { return right_pivots_; }
    size_t get_splits() const
    { return splits_; }
    vector<NSpillTreeNode *> get_children() const
    { return children_; }
    vector<size_t> get_domain() const
    { return domain_; }
    vector<double> tie_breaker() const
    { return tie_breaker_; }
    vector<double> tie_left_pivots() const
    { return tie_left_pivots_; }
    vector<double> tie_right_pivots() const
    { return tie_right_pivots_; }
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
    vector<T> left_pivots, right_pivots;
    size_t child_size = (size_t)(values.size() / splits);
    size_t spill_size = (size_t)(values.size() * spill_factor);
    LOG_FINE("> lim = %ld\n", child_size);
    for (int i=1; i<splits; i++) { // only (#_of_splits - 1) pivots
        pivots_left.push_back(selector(values, child_size * i - spill_size));
    }
    for (int i=1; i<splits; i++) { // only (#_of_splits - 1) pivots
        pivots_right.push_back(selector(values, child_size * i + spill_size));
    }
    
    
    //split the values into groups
    vector<vector<size_t>> left_pivot_pools(splits-1);
    vector<vector<size_t>> right_pivot_pools(splits-1);
    vector<vector<size_t>> children(splits);
    for (int i=0; i<domain.size(); i++) {
        bool not_pushed = true;
        int j = 0;
        while(not_pushed && j<splits-1) {
            if (values[i] == left_pivots[j]) {
                //if value equal to pivot, push into pivot pool
                left_pivot_pools[j].push_back(domain[i]);
                not_pushed = false;
            }
            else if (values[i] == right_pivots[j]) {
                right_pivot_pools[j].push_back(domain[i]);
                not_pushed = false;
            }
            else if (values[i] < left_pivots[j]) {
                children[j].push_back(domain[i]);
                not_pushed = false;
            }
            else if (left_pivots[j] < values[i] && values[i] < right_pivots[j]) {
                children[j].push_back(domain[i]);
                children[j+1].push_back(domain[i]);
                not_pushed = false;
            }
            j++;
        }
        if (not_pushed) {
            children[splits-1].push_back(domain[i]);
        }
    }
    
    //distribute pivot pool to all the children nodes
    //dot a random vector then do split again
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    //store new pivots in update_pool
    vector<double> tie_left_pivots(splits-1);
    vector<double> tie_right_pivots(splits-1);
    for (int i = 0; i < left_pivot_pools.size(); i++) {
        //extract the vectors from dataset
        DataSet<Label, T> tie_vectors = st.subset(left_pivot_pools[i]);
        //update pool using randome tie breaker
        vector<double> update_pool;
        for (int j = 0; j < tie_vectors.size(); j++) {
            double product = dot(*tie_vectors[j], tie_breaker);
            update_pool.push_back(product);
        }
        //find the tie_pivots and distribute tie vectors
        int filled_size = 0;
        int pivot_pool_i = i;
        i -= 1;
        while (update_pool.size() > filled_size) {
            i += 1;
            int k_left = child_size - children[i].size() + filled_size;
            //current pivot pool won't fill the next child
            //throw all the rest of this pivot pool into the next child
            if (k > update_pool.size()) {
                for (int j = 0; j < update_pool.size(); j++) {
                    if (update_pool[j] > tie_pivots[i-1]) {
                        children[i].push_back(pivot_pools[pivot_pool_i][j]);
                        filled_size++;
                    }
                }
                continue;
            }
            //current pivot pool will fill the child
            tie_pivots[i] = selector(update_pool, k);
            for (int j = 0; j < update_pool.size(); j++) {
                if (i == 0 && update_pool[j] <= tie_pivots[i]) {
                    children[i].push_back(pivot_pools[pivot_pool_i][j]);
                    filled_size++;
                }
                else if (update_pool[j] <= tie_pivots[i] && update_pool[j] > tie_pivots[i-1]) {
                    children[i].push_back(pivot_pools[pivot_pool_i][j]);
                    filled_size++;
                }
                else if (i == tie_pivots.size() && update_pool[j] > tie_pivots[i-1]) {
                    children[i].push_back(pivot_pools[pivot_pool_i][j]);
                    filled_size++;
                }
            }
            //children[i] should be full now
        }
    }

    
    //call build tree recursively to build tree
    vector<NSpillTreeNode<Label, T> *> children_vector;
    for (int i=0; i<splits; i++) {
        NSpillTreeNode<Label, T> * node = build_tree(leaf_size, st, splits, spill_factor, children[i]);
        children_vector.push_back(node);
    }
    //return newly built tree
    NSpillTreeNode<Label, T> * result = new NSpillTreeNode<Label, T>(dimension, mx_var_index, pivots, splits, tie_breaker, tie_pivots, children_vector, domain);
    LOG_INFO("Exit build_tree\n");
    return result;
}

/* Public Functions */

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(const vector<size_t> domain) :
  dimension_ (0),
  index_ (0),
  pivots_ (NULL),
  splits_ (0),
  tie_breaker_(NULL),
  tie_pivots_(NULL),
  children_ (NULL),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(
        size_t dimension,
		size_t index,
        vector<T> pivots,
		size_t splits,
        vector<double> tie_breaker,
        vector<double> tie_pivots,
		vector<NSpillTreeNode *> children, 
		vector<size_t> domain) :
  dimension_ (dimension),
  index_ (index),
  pivots_ (pivots),
  splits_ (splits),
  tie_breaker_(tie_breaker),
  tie_pivots_(tie_pivots),
  children_ (children),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with index = %ld, domain.size = %ld\n", index, 
            domain.size());
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
    while (number_of_pivots--)
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
    number_of_pivots = splits_ - 1;
    if (splits_ == 0)
        number_of_pivots = 0;
    while (number_of_pivots--)
    {
        double tie_pivot;
        in.read((char *)&tie_pivot, sizeof(double));
        tie_pivots_.push_back(tie_pivot);
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
    out.write((char *)&tie_pivots_[0],
              sizeof(double) * tie_pivots_.size());
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
            bool pushed = false; //To indicate whether it is pushed to any child
            size_t splits = cur->splits_;
            bool tie = false; //To indicate whether it equals to any pivots
            for (int i = 0; i < splits-1; i++) {
                if ((*query)[cur->index_] < cur->pivots_[i]) {
                    expl.push(cur->children_[i]);
                    pushed = true;
                    break;
                }
                else if ((*query)[cur->index_] == cur->pivots_[i]) {
                    tie = true;
                    vector<double> tie_breaker = cur->tie_breaker_;
                    double product = dot(*query, tie_breaker);
                    if (product <= cur->tie_pivots_[i]) {
                        expl.push(cur->children_[i]);
                        pushed = true;
                        break;
                    }
                }
                //Push to the current child if it equlas to the previous pivots
                //but not pushed into the previous child
                else if (tie) {
                    expl.push(cur->children_[i]);
                    pushed = true;
                    break;
                }
            }
            if (!pushed)
                expl.push(cur->children_[splits-1]);
        }
        else
            return cur->domain_;
    }
    LOG_INFO("Exit subdomain\n");
    return vector<size_t>();
}

#endif
