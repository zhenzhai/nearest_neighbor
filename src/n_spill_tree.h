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
 *                    children_ - Array of subtree nodes
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
    vector<size_t> pivots_;
    size_t splits_;
    vector<NSpillTreeNode> children_;
    vector<size_t> domain_;
public:
    NSpillTreeNode(const vector<size_t> domain);
    NSpillTreeNode(size_t index, vector<size_t> pivots, size_t splits, vector<NSpillTreeNode> children, vector<size_t> domain);
    //NSpillTreeNode(ifstream & in);
    ~NSpillTreeNode();
    size_t get_index() const
    { return index_; }
    vector<size_t> get_pivots() const
    { return pivots_; }
    size_t get_splits() const
    { return splits_; }
    vector<NSpillTreeNode> get_children() const
    { return children_; }
    vector<size_t> get_domain() const
    { return domain_; }
    /*void set_children(vector<NSpillTreeNode> children)
    { children_ = children; };*/
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
 *                          - Creates a tree of given min leaf size and
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
    static NSpillTreeNode<Label, T> * build_tree(size_t c,
            DataSet<Label, T> & st, size_t splits, double a, vector<size_t> domain);
protected:
    NSpillTreeNode<Label, T> * root_;
    DataSet<Label, T> & st_;
    NSpillTree(DataSet<Label, T> & st);
public:
    NSpillTree(size_t c, double a, DataSet<Label, T> & st);
    NSpillTree(ifstream & in, DataSet<Label, T> & st);
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

/* Private Functions */

template<class Label, class T>
NSpillTreeNode<Label, T> * NSpillTree<Label, T>::build_tree(size_t c,
        DataSet<Label, T> & st, size_t splits, double a, vector<size_t> domain)
{
    LOG_INFO("Enter build_tree\n");
    LOG_FINE("with c = %ld, splits = %ld, and domain size = %ld\n", c, splits, domain.size());
    if (domain.size() < c) {
        LOG_INFO("Exit build_tree");
        LOG_FINE("by hitting base size");
        return new NSpillTreeNode<Label, T>(domain);
    }
    DataSet<Label, T> subst = st.subset(domain);
    size_t mx_var_index = max_variance_index(subst);
    //get all the values at the max variance index
    vector<T> values;
    for (size_t i = 0; i < subst.size(); i++) {
        values.push_back((*subst[i])[mx_var_index]);
    }
    vector<size_t> children_array;
    //create and calculate the split values
    vector<size_t> pivots;
    size_t subdomain_lim = (size_t)(values.size() / splits) + (a * domain.size());
    LOG_FINE("> lim = %ld\n", subdomain_lim);
    for (int i=1; i<splits; i++) {
        pivots.push_back(selector(values, (size_t)(values.size() / splits) * i));
    }
    //split the values into groups
    vector<size_t> * pivots_pool = new vector<size_t>[splits];
    vector<size_t> * children = new vector<size_t>[splits];
    for (int i=0; i<domain.size(); i++) {
        bool not_pushed = true;
        for (int j=0; j<splits-1; j++) {
            if (values[i] < pivots[j]) {
                children[j].push_back(domain[i]);
                not_pushed = false;
            }
            else if (values[i] == pivots[j]) {
                pivots_pool[j].push_back(domain[i]);
                not_pushed = false;
            }
        }
        if (not_pushed) {
            children[splits-1].push_back(domain[i]);
        }
    }
    //children = break_tie(subdomain_lim, children, pivots_pool);
    //call build tree recursively to build tree
    for (int i=0; i<splits; i++) {
        children_array.push_back(build_tree(c, st, splits, a, children[i]));
    }
    //return newly built tree
    NSpillTreeNode<Label, T> * result = new NSpillTreeNode<Label, T>(mx_var_index, pivots, children_array, domain);
    LOG_INFO("Exit build_tree\n");
    return result;
}

/* Public Functions */

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(const vector<size_t> domain) :
  index_ (0),
  pivots_ (NULL),
  splits_ (0),
  children_ (NULL),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with domain.size = %ld\n", domain.size());
}

template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(size_t index,
        vector<size_t> pivots, size_t splits, vector<NSpillTreeNode> children, vector<size_t> domain) :
  index_ (index),
  pivots_ (pivots),
  splits_ (splits),
  children_ (children),
  domain_ (domain)
{ 
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with index = %ld, domain.size = %ld\n", index, 
            domain.size());
}

/*template<class Label, class T>
NSpillTreeNode<Label, T>::NSpillTreeNode(ifstream & in)
{
    LOG_INFO("NSpillTreeNode Constructed\n");
    LOG_FINE("with input stream\n");
    in.read((char *)&index_, sizeof(size_t));
    in.read((char *)&splits_, sizeof(size_t));
    size_t number_of_pivots = splits_ - 1;
    while (number_of_pivots--)
    {
        size_t pivot;
        in.read((char *)&pivot, sizeof(T));
        pivots_.push_back(pivot);
    }
    size_t sz;
    LOG_FINE("> sz = %ld\n", sz);
    in.read((char *)&sz, sizeof(size_t));
    while (sz--)
    {
        size_t v;
        in.read((char *)&v, sizeof(size_t));
        domain_.push_back(v);
    }
}*/

template<class Label, class T>
NSpillTreeNode<Label, T>::~NSpillTreeNode()
{
    LOG_FINE("Deleted children\n");
    delete children_;
    LOG_INFO("NSpillTreeNode Deconstructed\n");
}

template<class Label, class T>
void NSpillTreeNode<Label, T>::save(ofstream & out) const
{
    LOG_INFO("Saving NSpillTreeNode\n");
    LOG_FINE("> domain.size = %ld\n", domain_.size());
    out.write((char *)&index_, sizeof(size_t));
    out.write((char *)&splits_, sizeof(size_t));
    out.write((char *)&pivots_[0],
              sizeof(size_t) * pivots_.size());
    size_t sz = domain_.size();
    out.write((char *)&sz, sizeof(size_t)); 
    out.write((char *)&domain_[0],
              sizeof(size_t) * domain_.size());
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
NSpillTree<Label, T>::NSpillTree(size_t c, double a, DataSet<Label, T> & st) :
  root_ (build_tree(c, st, st.get_splits(), a, st.get_domain())),
  st_ (st)
{ 
    LOG_INFO("NSpillTree Constructed\n");
    LOG_FINE("with c = %ld, a = %lf\n", c, a);
}

template<class Label, class T>
NSpillTree<Label, T>::NSpillTree(ifstream & in, DataSet<Label, T> & st) :
  st_ (st)
{
    LOG_INFO("NSpillTree Constructed\n");
    LOG_FINE("with input stream\n");
    queue<NSpillTreeNode<Label, T> **> to_load;
    to_load.push(&root_);
    int splits = root_->splits_;
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
        NSpillTreeNode<Label, T>** ary_children = new NSpillTreeNode<Label, T>* [splits];
        for (int i=0; i<splits; i++){
            to_load.push(ary_children + i);
        }
        
        (*cur)->children_.assign(ary_children, ary_children + splits);
        delete [] ary_children;
        *cur = new NSpillTreeNode<Label, T>(in);
        for (int i=0; i<splits; i++){
            to_load.push(&(*cur)->children_[i]);
        }
    }
}

template<class Label, class T>
NSpillTree<Label, T>::~NSpillTree()
{
    if (root_) delete root_;
    LOG_INFO("NSpillTree Deconstructed\n");
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
            for (int i=0; i<cur->splits_-1; i++) {
                to_save.push(cur->children_[i]);
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
        if (cur->children_ &&
            cur->domain_.size() >= l_c) {
            bool pushed = false;
            int splits = cur->splits;
            for (int i=0; i<splits-1; i++) {
                if ((*query)[cur->index_] <= cur->pivots_[i]) {
                    expl.push(cur->children_[i]);
                    pushed = true;
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
