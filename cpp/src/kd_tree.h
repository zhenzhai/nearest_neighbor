/* 
 * File             : kd_tree.h
 * Summary          : Infrastructure to hold a kd tree.
 */
#ifndef KD_TREE_H_
#define KD_TREE_H_

#include <queue>
#include <map>
#include "logging.h"
#include "data_set.h"
#include <errno.h>
using namespace std;

/* Class Prototypes */

template<class Label, class T>
class KDTreeNode;

template<class Label, class T>
class KDTree;

/* Class Definitions */

/* 
 * Name             : KDTreeNode
 * Description      : Data structure to hold a node of a KDTree
 * Data Field(s)    : index_        - The max variance index
 *                    pivot_        - The value of pivot
 *                    tie_breaker_  - The tie breaker vector
 *                    tie_pivot_    - The value of pivot of tie breaker
 *                    dimension_    - The dimension of feature, used in de-serialization
 *                    left_         - Pointer to left subtree node
 *                    right_        - Pointer to right subtree node
 *                    domain_       - vectors out of vector space in data set
 * Functions(s)     : KDTreeNode(const vector<size_t>) 
 *                              - Create a KDTreeNode of given domain (leaf)
 *                    KDTreeNode(size_t, T, vector<size_t>)
 *                              - Create a KDTreeNode of given domain (non-leaf)
 *                    KDTreeNode(ifstream &)
 *                              - Creates a KDTreeNode through de-serialization
 *                    size_t get_index() const
 *                              - Gets index of max variance
 *                    KDTreeNode * get_left() const
 *                              - Returns pointer to left subtree node
 *                    KDTreeNode * get_right() const
 *                              - Returns pointer to right subtree node
 *                    vector<size_t> get_domain() const
 *                              - Returns the domain the node stores
 *                    void set_left(KDTreeNode *)
 *                              - Sets the left subtree node
 *                    void set_right(KDTreeNode *)
 *                              - Sets the right subtree node
 *                    void save(ofstream &)
 *                              - Serializes node 
 */
template<class Label, class T>
class KDTreeNode
{
protected:
    size_t index_;
    T pivot_;
    vector<double> tie_breaker_;
    double tie_pivot_;
    size_t dimension_;
    KDTreeNode * left_, * right_;
    vector<size_t> domain_;
public:
    KDTreeNode(const vector<size_t> domain);
    KDTreeNode(size_t index, T pivot, vector<size_t> domain, size_t dimension, double tie_pivot, vector<double> tie_breaker);
    KDTreeNode(ifstream & in);
    ~KDTreeNode();
    size_t get_index() const
    { return index_; }
    KDTreeNode * get_left() const
    { return left_; }
    KDTreeNode * get_right() const
    { return right_; }
    T get_pivot() const
    { return pivot_;}
    vector<size_t> get_domain() const
    { return domain_; }
    void set_left(KDTreeNode * left)
    { left_ = left; };
    void set_right(KDTreeNode * right)
    { right_ = right; };
    virtual void save(ofstream & out) const;
    friend class KDTree<Label, T>;
};

/*
 * Name             : KDTree
 * Description      : Encapsulates the KDTreeNodes into tree.
 * Data Field(s)    : root_ - Holds the root node of tree
 *                    st_   - Holds the data set associated with tree
 * Function(s)      : KDTree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    KDTree(size_t, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size and
 *                            data set
 *                    KDTree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization 
 *                    ~KDTree()
 *                          - Deconstructor
 *                    KDTreeNode<Label, T> * get_root() const
 *                          - Returns the root
 *                    DataSet<Label, T> & get_st() const
 *                          - Returns the set associated with the tree
 *                    void save(ofstream &) const
 *                          - Serializes the tree
 *                    vector<size_t> subdomain(vector<T> *, size_t)
 *                          - Queries the tree for a subdomain
 */
template<class Label, class T>
class KDTree
{
private:
    static KDTreeNode<Label, T> * build_tree(size_t min_leaf_size,
            DataSet<Label, T> & st, vector<size_t> domain);
protected:
    KDTreeNode<Label, T> * root_;
    DataSet<Label, T> & st_;
    KDTree(DataSet<Label, T> & st);
public:
    KDTree(size_t min_leaf_size, DataSet<Label, T> & st);
    KDTree(ifstream & in, DataSet<Label, T> & st);
    ~KDTree();
    KDTreeNode<Label, T> * get_root() const
    { return root_; }
    DataSet<Label, T> & get_st() const
    { return st_; }
    void set_root(KDTreeNode<Label, T> * root)
    { root_ = root; }
    virtual void save(ofstream & out) const;
    virtual vector<size_t> subdomain(vector<T> * query, size_t l_c = 0);
};

/* Private Functions */

template<class Label, class T>
KDTreeNode<Label, T> * KDTree<Label, T>::build_tree(size_t min_leaf_size,
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
    size_t mx_var_index = max_variance_index(subst);
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++) {
        values.push_back((*subst[i])[mx_var_index]);
    }
    double pivot = selector(values, (size_t)(values.size() * 0.5));
    vector<size_t> subdomain_l;
    size_t subdomain_l_lim = (size_t)(values.size() * 0.5);
    LOG_FINE("> left_limit = %ld\n", subdomain_l_lim);
    vector<size_t> subdomain_r;
    vector<size_t> pivot_pool;
    for (size_t i = 0; i < domain.size(); i++) {
        if (values[i] == pivot)
            pivot_pool.push_back(domain[i]);
        else if (values[i] < pivot)
            subdomain_l.push_back(domain[i]);
        else
            subdomain_r.push_back(domain[i]);
    }
    
    //distribute pivot pool to all the children nodes
    //dot a random vector then do split again
    size_t dimension = (*subst[0]).size();
    vector<double> tie_breaker = random_tie_breaker(dimension);
    
    //extract the vectors from dataset
    DataSet<Label, T> tie_vectors = st.subset(pivot_pool);
    
    //update pool using randome tie breaker
    vector<double> update_pool;
	double product = 0;
    for (int j = 0; j < tie_vectors.size(); j++) {
		vector<T> tmp = *tie_vectors[j];
        product = dot(*tie_vectors[j], tie_breaker);
        update_pool.push_back(product);
		product = 0;
    }
    
    //find the tie_pivots and distribute tie vectors
    int k = subdomain_l_lim - subdomain_l.size();
    
    //store new pivots in update_pool
    double tie_pivot;
    
    tie_pivot = selector(update_pool, k);
    for (int j = 0; j < update_pool.size(); j++) {
        if (update_pool[j] <= tie_pivot)
            subdomain_l.push_back(pivot_pool[j]);
        else
            subdomain_r.push_back(pivot_pool[j]);
    }

    /*while (subdomain_l_lim > subdomain_l.size()) {
        size_t curr = pivot_pool.back();
        pivot_pool.pop_back();
        subdomain_l.push_back(curr);
    }
    while (!pivot_pool.empty()) {
        size_t curr = pivot_pool.back();
        pivot_pool.pop_back();
        subdomain_r.push_back(curr);
    }*/
    
    KDTreeNode<Label, T> * result = new KDTreeNode<Label, T>
            (mx_var_index, pivot, domain, dimension, tie_pivot, tie_breaker);
    result->left_ = build_tree(min_leaf_size, st, subdomain_l);
    result->right_ = build_tree(min_leaf_size, st, subdomain_r);
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_FINE("Exit build_tree\n");
    return result;
}

/* Public Functions */

template<class Label, class T>
KDTreeNode<Label, T>::KDTreeNode(const vector<size_t> domain) :
  index_ (0),
  pivot_ (0),
  tie_pivot_(0),
  dimension_(0),
  tie_breaker_(NULL),
  left_ (NULL),
  right_ (NULL),
  domain_ (domain)
{ 
    LOG_FINE("KDTreeNode Constructed\n"); 
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
KDTreeNode<Label, T>::KDTreeNode(size_t index, 
        T pivot, vector<size_t> domain, size_t dimension,
        double tie_pivot, vector<double> tie_breaker) :
  index_ (index),
  pivot_ (pivot),
  tie_pivot_(tie_pivot),
  tie_breaker_(tie_breaker),
  dimension_ (dimension),
  left_ (NULL), 
  right_ (NULL),
  domain_ (domain)
{ 
    LOG_FINE("KDTreeNode Constructed\n"); 
    LOG_FINE("with index = %ld, domain.size = %ld\n", index, 
            domain.size());
}

template<class Label, class T>
KDTreeNode<Label, T>::KDTreeNode(ifstream & in)
{
    LOG_FINE("KDTreeNode Constructed\n"); 
    LOG_FINE("with input stream\n");
    in.read((char *)&index_, sizeof(size_t));
    in.read((char *)&pivot_, sizeof(T));
    in.read((char *)&tie_pivot_, sizeof(double));
    in.read((char *)&dimension_, sizeof(size_t));
    size_t dimen = dimension_;
    LOG_FINE("> dimension = %ld\n", dimen);
    while (dimen--)
    {
        double v = 0;
        in.read((char *)&v, sizeof(double));
        tie_breaker_.push_back(v);
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
}

template<class Label, class T>
KDTreeNode<Label, T>::~KDTreeNode()
{
    if (left_) {
        LOG_FINE("Deleted left subtree\n");
        delete left_;
    }
    if (right_) {
        LOG_FINE("Deleted right subtree\n");
        delete right_;
    }
    LOG_FINE("KDTreeNode Deconstructed\n"); 
}

template<class Label, class T>
void KDTreeNode<Label, T>::save(ofstream & out) const
{
    LOG_FINE("Saving KDTreeNode\n"); 
    LOG_FINE("> domain.size = %ld\n", domain_.size());
    out.write((char *)&index_, sizeof(size_t)); 
    out.write((char *)&pivot_, sizeof(T));
    out.write((char *)&tie_pivot_,sizeof(double));
    out.write((char *)&dimension_, sizeof(size_t));
	if (tie_breaker_.empty()){
		out.write((char *)&tie_breaker_, sizeof(double) * dimension_);
	} else {
		out.write((char *)&tie_breaker_[0], sizeof(double) * dimension_);
	}
    size_t sz = domain_.size();
    out.write((char *)&sz, sizeof(size_t)); 
    out.write((char *)&domain_[0], 
            sizeof(size_t) * domain_.size());
}

template<class Label, class T>
KDTree<Label, T>::KDTree(DataSet<Label, T> & st) :
  root_ (NULL),
  st_ (st)
{ 
    LOG_INFO("KDTree Constructed\n"); 
    LOG_FINE("with default constructor\n");
}

template<class Label, class T>
KDTree<Label, T>::KDTree(size_t min_leaf_size, DataSet<Label, T> & st) :
  root_ (build_tree(min_leaf_size, st, st.get_domain())),
  st_ (st)
{ 
    LOG_INFO("KDTree Constructed\n"); 
    LOG_FINE("with min_leaf_size = %ld", min_leaf_size);
}

template<class Label, class T>
KDTree<Label, T>::KDTree(ifstream & in, DataSet<Label, T> & st) :
  st_ (st)
{
    LOG_INFO("KDTree Constructed\n");
    LOG_FINE("with input stream\n");
    queue<KDTreeNode<Label, T> **> to_load;
    to_load.push(&root_);
    while (!to_load.empty())
    {
        KDTreeNode<Label, T> ** cur = to_load.front();
        to_load.pop();
        bool exist;
        in.read((char *)&exist, sizeof(bool));
        if (!exist) {
            *cur = NULL; 
            continue;
        }
        *cur = new KDTreeNode<Label, T>(in);
        to_load.push(&(*cur)->left_);
        to_load.push(&(*cur)->right_);
    }
}

template<class Label, class T>
KDTree<Label, T>::~KDTree()
{
    if (root_) delete root_;
    LOG_INFO("KDTree Deconstructed\n"); 
}

template<class Label, class T>
void KDTree<Label, T>::save(ofstream & out) const
{
    LOG_INFO("Saving KDTreeNode\n"); 
    queue<KDTreeNode<Label, T> *> to_save;
    to_save.push(root_);
    while (!to_save.empty()) {
        KDTreeNode<Label, T> * cur = to_save.front();
        to_save.pop();
        bool exists = cur != NULL;
        out.write((char *)&exists, sizeof(bool)); 
        if (exists) {
            cur->save(out);
            to_save.push(cur->left_);
            to_save.push(cur->right_);
        }
    }
}

template<class Label, class T>
vector<size_t> KDTree<Label, T>::subdomain(vector<T> * query, size_t l_c)
{
    LOG_FINE("Enter subdomain\n");
    LOG_FINE("with lc = %ld\n", l_c);
    queue<KDTreeNode<Label, T> *> expl;
    expl.push(root_);
    while (!expl.empty())
    {
        KDTreeNode<Label, T> * cur = expl.front();
        expl.pop();
        if (cur->left_ && cur->right_ &&
            cur->domain_.size() >= l_c) {
            if ((*query)[cur->index_] < cur->pivot_)
                expl.push(cur->left_);
            else if ((*query)[cur->index_] > cur->pivot_)
                expl.push(cur->right_);
            else {
                double product = dot(*query, cur->tie_breaker_);
                if (product <= cur->tie_pivot_)
                    expl.push(cur->left_);
                else
                    expl.push(cur->right_);
            }
        }
        else
            return cur->domain_;
    }
    LOG_FINE("Exit subdomain\n");
    return vector<size_t>();
}

#endif
