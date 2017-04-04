/* 
 * File             : pca_tree.h
 * Summary          : Infrastructure to hold a binary space partition tree.
 */
#ifndef PCA_TREE_H_
#define PCA_TREE_H_

#include <queue>
#include <map>
#include "vector_math.h"
#include "data_set.h"
using namespace std;

/* Class Prototypes */
template<class Label, class T>
class PCATreeNode;

template<class Label, class T>
class PCATree;

/* Class Definitions */


/* 
 * Name             : PCATreeNode
 * Description      : Data structure to hold a node of a KDTree
 * Data Field(s)    : dir_      - Direction to project the distance onto
 *                    pivot_    - The value to pivot on
 *                    left_     - Pointer to left subtree node
 *                    right_    - Pointer to right subtree node
 *                    domain_   - vectors out of vector space in data set
 * Functions(s)     : PCATreeNode(const vector<size_t>) 
 *                              - Create a PCATreeNode of given domain (leaf)
 *                    PCATreeNode(size_t, T, vector<size_t>)
 *                              - Create a PCATreeNode of given domain (non-leaf)
 *                    PCATreeNode(ifstream &)
 *                              - Creates a KDTreeNode through de-serialization
 *                    vector<double> get_direction() const
 *                              - Returns the direction to project the distance onto
 *                    size_t get_index() const
 *                              - Gets index of max variance
 *                    PCATreeNode * get_left() const
 *                              - Returns pointer to left subtree node
 *                    PCATreeNode * get_right() const
 *                              - Returns pointer to right subtree node
 *                    vector<size_t> get_domain() const
 *                              - Returns the domain the node stores
 *                    void set_left(PCATreeNode *)
 *                              - Sets the left subtree node
 *                    void set_right(PCATreeNode *)
 *                              - Sets the right subtree node
 *                    void save(ofstream &)
 *                              - Serializes node 
 */
template<class Label, class T>
class PCATreeNode
{
protected:
    PCATreeNode * left_, * right_;
    vector<double> dir_;
    double pivot_;
    vector<size_t> domain_;
public:
    PCATreeNode(const vector<size_t> domain);
    PCATreeNode(vector<double> dir, double pivot, vector<size_t> domain);
    PCATreeNode(ifstream & in);
    ~PCATreeNode();
    vector<double> get_direction() const
    { return dir_; }
    double get_pivot() const
    { return pivot_; }
    PCATreeNode * get_left() const
    { return left_; }
    PCATreeNode * get_right() const
    { return right_; }
    vector<size_t> get_domain() const
    { return domain_; }
    void set_left(PCATreeNode * left)
    { left_ = left; };
    void set_right(PCATreeNode * right)
    { right_ = right; };
    virtual void save(ofstream & out) const;
    friend class PCATree<Label, T>;
};

/*
 * Name             : PCATree
 * Description      : Encapsulates the PCATreeNodes into tree.
 * Data Field(s)    : root_ - Holds the root node of tree
 *                    st_   - Holds the data set associated with tree
 * Function(s)      : PCATree(DataSet<Label, T>)
 *                          - Creates a tree of given data set
 *                    PCATree(size_t, DataSet<Label, T>)
 *                          - Creates a tree of given min leaf size and
 *                            data set
 *                    PCATree(ifstream &, DataSet<Label, T>)
 *                          - De-serialization 
 *                    ~PCATree()
 *                          - Deconstructor
 *                    PCATreeNode<Label, T> * get_root() const
 *                          - Returns the root
 *                    DataSet<Label, T> & get_st() const
 *                          - Returns the set associated with the tree
 *                    void save(ofstream &) const
 *                          - Serializes the tree
 *                    vector<size_t> subdomain(vector<T> *, size_t)
 *                          - Queries the tree for a subdomain
 */
template<class Label, class T>
class PCATree
{
private:
    static PCATreeNode<Label, T> * build_tree(size_t c,
            DataSet<Label, T> & st, vector<size_t> domain);
protected:
    PCATreeNode<Label, T> * root_;
    DataSet<Label, T> & st_;
public:
    PCATree(DataSet<Label, T> & st);
    PCATree(size_t min_leaf_size, DataSet<Label, T> & st);
    PCATree(ifstream & in, DataSet<Label, T> & st);
    ~PCATree();
    PCATreeNode<Label, T> * get_root() const
    { return root_; }
    DataSet<Label, T> & get_st() const
    { return st_; }
    void set_root(PCATreeNode<Label, T> * root)
    { root_ = root; }
    virtual void save(ofstream & out) const;
    virtual vector<size_t> subdomain(vector<T> * query, size_t l_c = 0);
};

template<class Label, class T>
PCATreeNode<Label, T> * PCATree<Label, T>::build_tree(size_t min_leaf_size,
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

	/*find pivot to split*/
    double pivot = selector(values, (size_t)(values.size() * 0.5));

	/*split to left and right child*/
    vector<size_t> subdomain_l;
    size_t subdomain_l_lim = (size_t)(values.size() * 0.5);
    LOG_FINE("> left_lim = %ld\n", subdomain_l_lim);
    vector<size_t> subdomain_r;
    for (size_t i = 0; i < domain.size(); i++) {
        if (values[i] <= pivot)
            subdomain_l.push_back(domain[i]);
        else
            subdomain_r.push_back(domain[i]);
    }

    PCATreeNode<Label, T> * result = new PCATreeNode<Label, T>(mx_var_dir, 
            pivot, domain);
    result->left_ = build_tree(min_leaf_size, st, subdomain_l);
    result->right_ = build_tree(min_leaf_size, st, subdomain_r);
    LOG_FINE("> sdl = %ld\n", subdomain_l.size());
    LOG_FINE("> sdr = %ld\n", subdomain_r.size());
    LOG_FINE("Exit build_tree\n");
    return result;
}

template<class Label, class T>
PCATreeNode<Label, T>::PCATreeNode(const vector<size_t> domain) :
  dir_ (),
  pivot_ (0),
  left_ (NULL),
  right_ (NULL),
  domain_ (domain)
{ 
    LOG_FINE("PCATreeNode Constructed\n"); 
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
PCATreeNode<Label, T>::PCATreeNode(vector<double> dir, 
        double pivot, vector<size_t> domain) :
  dir_ (dir),
  pivot_ (pivot),
  left_ (NULL), 
  right_ (NULL),
  domain_ (domain)
{ 
    LOG_FINE("PCATreeNode Constructed\n"); 
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
PCATreeNode<Label, T>::PCATreeNode(ifstream & in)
{
    LOG_FINE("PCATreeNode Constructed\n"); 
    LOG_FINE("with input stream\n");
    size_t dim;
    in.read((char *)&dim, sizeof(size_t));
    while (dim--) {
        double v;
        in.read((char *)&v, sizeof(double));
        dir_.push_back(v);
    }
    in.read((char *)&pivot_, sizeof(double));
    size_t sz;
    in.read((char *)&sz, sizeof(size_t));
    while (sz--) {
        size_t v;
        in.read((char *)&v, sizeof(size_t));
        domain_.push_back(v);
    }
}

template<class Label, class T>
PCATreeNode<Label, T>::~PCATreeNode()
{
    if (left_) {
        LOG_FINE("Deleted left subtree\n");
        delete left_;
    }
    if (right_) {
        LOG_FINE("Deleted right subtree\n");
        delete right_;
    }
    LOG_FINE("PCATreeNode Deconstructed\n"); 
}

template<class Label, class T>
void PCATreeNode<Label, T>::save(ofstream & out) const
{
    LOG_FINE("Saving PCATreeNode\n"); 
    LOG_FINE("> domain.size = %ld\n", domain_.size());
    size_t dim = dir_.size();
    out.write((char *)&dim, sizeof(size_t));
	if (dir_.empty()) {
		out.write((char *)&dir_, sizeof(double) * dim);
	}
	else {
		out.write((char *)&dir_[0], sizeof(double) * dim);
	}
    out.write((char *)&pivot_, sizeof(double)); 
    size_t sz = domain_.size();
    out.write((char *)&sz, sizeof(size_t)); 
    out.write((char *)&domain_[0], 
            sizeof(size_t) * domain_.size());
}

template<class Label, class T>
PCATree<Label, T>::PCATree(DataSet<Label, T> & st) :
  root_ (NULL),
  st_ (st)
{ 
    LOG_INFO("PCATree Constructed\n");
    LOG_FINE("with default constructor\n");
}

template<class Label, class T>
PCATree<Label, T>::PCATree(size_t min_leaf_size, DataSet<Label, T> & st) :
  root_ (build_tree(min_leaf_size, st, st.get_domain())),
  st_ (st)
{ 
    LOG_INFO("PCATree Constructed\n");
    LOG_FINE("with min_leaf_size = %ld", min_leaf_size);
}

template<class Label, class T>
PCATree<Label, T>::PCATree(ifstream & in, DataSet<Label, T> & st) :
  st_ (st)
{
    LOG_INFO("PCATree Constructed\n");
    LOG_FINE("with input stream\n");
    queue<PCATreeNode<Label, T> **> to_load;
    to_load.push(&root_);
    while (!to_load.empty()) {
        PCATreeNode<Label, T> ** cur = to_load.front();
        to_load.pop();
        bool exist;
        in.read((char *)&exist, sizeof(bool));
        if (!exist) {
            *cur = NULL; 
            continue;
        }
        *cur = new PCATreeNode<Label, T>(in);
        to_load.push(&(*cur)->left_);
        to_load.push(&(*cur)->right_);
    }
}

template<class Label, class T>
PCATree<Label, T>::~PCATree()
{
    if (root_) delete root_;
    LOG_INFO("PCATree Deconstructed\n"); 
}

template<class Label, class T>
void PCATree<Label, T>::save(ofstream & out) const
{
    LOG_FINE("Saving PCATreeNode\n"); 
    queue<PCATreeNode<Label, T> *> to_save;
    to_save.push(root_);
    while (!to_save.empty()) {
        PCATreeNode<Label, T> * cur = to_save.front();
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
vector<size_t> PCATree<Label, T>::subdomain(vector<T> * query, size_t leaf_size)
{
    LOG_FINE("Enter subdomain\n");
    LOG_FINE("with leaf_size = %ld\n", leaf_size);
    queue<PCATreeNode<Label, T> *> expl;
    expl.push(root_);
    while (!expl.empty()) {
        PCATreeNode<Label, T> * cur = expl.front();
        expl.pop();
        if (cur->left_ && cur->right_ &&
            cur->domain_.size() >= leaf_size) {
            if (dot(*query, cur->dir_) <= cur->pivot_)
                expl.push(cur->left_);
            else
                expl.push(cur->right_);
        }
        else
            return cur->domain_;
    }
    LOG_FINE("Exit subdomain\n");
    return vector<size_t>();
}
#endif
