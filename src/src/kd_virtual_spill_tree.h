/* 
 * File             : kd_spill_tree.h
 * Date             : 2014-5-29
 * Summary          : Infrastructure to hold a kd spill tree.
 */
#ifndef KD_VIRTUAL_SPILL_TREE_H_
#define KD_VIRTUAL_SPILL_TREE_H_

#include <map>
#include <queue>
#include <set>
#include <utility>
#include "kd_tree.h"
using namespace std;

/* Class Definitions */

/* 
 * Name             : KDVirtualSpillTree
 * Description      : Encapsulates the KDTreeNodes into a virtual spill tree.
 *                    Effectively acts as identically to KDTree with spillage
 *                    in terms of its query method.
 * Data Field(s)    : range_mp  - A map to match a node with its range
 * Functions(s)     : KDVirtualSpillTree(size_t, double, DataSet<Label, T> &)
 *                              - Creates a spill tree with given min leaf size
 *                                and the spill factor
 *                    KDVirtualSpillTree(ifStream & in, DataSet<Label, T> & st)
 *                              - De-serializes a virtual spill tree
 *                    void save(ofstream &) const 
 *                              - Serializes a virtual spill tree with its range
 *                                appended to the end
 *                    vector<size_t> subdomain(vector<T> *, size_t)
 *                              - Queries the node with the spillage
 */
template<class Label, class T>
class KDVirtualSpillTree : public KDTree<Label, T>
{
    typedef pair<T, T> range;
protected:
    map<KDTreeNode<Label, T> *, range> range_mp_;
    KDVirtualSpillTree(DataSet<Label, T> & st);
public:
    KDVirtualSpillTree(size_t c, double a, DataSet<Label, T> & st);
    KDVirtualSpillTree(ifstream & in, DataSet<Label, T> & st);
    virtual void save(ofstream & out) const;
    virtual vector<size_t> subdomain(vector<T> * query, size_t l_c = 0, size_t* l = 0);
};

/* Private Functions */

template<class Label, class T>
KDVirtualSpillTree<Label, T>::KDVirtualSpillTree(DataSet<Label, T> & st) :
  KDTree<Label, T>(st)
{ 
    LOG_INFO("KDVirtualSpillTree Constructed\n"); 
    LOG_FINE("with default constructor\n");
}

/* Public Functions */

template<class Label, class T>
KDVirtualSpillTree<Label, T>::KDVirtualSpillTree(size_t c, double a, 
        DataSet<Label, T> & st) :
  KDTree<Label, T>(c, st)
{
    LOG_INFO("KDVirtualSpillTree Constructed\n"); 
    LOG_FINE("with c = %ld, a = %lf\n", c, a);
    queue<KDTreeNode<Label, T> *> to_update;
    to_update.push((this->get_root()));
    while (!to_update.empty())
    {
        KDTreeNode<Label, T> * cur = to_update.front();
        to_update.pop();
        bool exists = cur != NULL;
        if (exists)
        {
            DataSet<Label, T> subst = st.subset(cur->get_domain());
            size_t mx_var_index = cur->get_index();
            vector<T> values;
            for (size_t i = 0; i < subst.size(); i++)
            {
                values.push_back((*subst[i])[mx_var_index]);
            }
            double pivot_l = selector(values, (size_t)(values.size() * (0.5 - a)));
            double pivot_r = selector(values, (size_t)(values.size() * (0.5 + a)));
            range_mp_[cur] = range(pivot_l, pivot_r);
            to_update.push(cur->get_left());
            to_update.push(cur->get_right());
        }
    }
}

template<class Label, class T>
KDVirtualSpillTree<Label, T>::KDVirtualSpillTree(ifstream & in, 
        DataSet<Label, T> & st) :
  KDTree<Label, T>(in, st)
{
    LOG_INFO("KDVirtualSpillTree Constructed\n"); 
    LOG_FINE("with input stream\n");
    queue<KDTreeNode<Label, T> *> to_update;
    to_update.push((this->get_root()));
    while (!to_update.empty())
    {
        KDTreeNode<Label, T> * cur = to_update.front();
        to_update.pop();
        bool exists = cur != NULL;
        if (exists)
        {
            T pivot_l, pivot_r;
            in.read((char *)&pivot_l, sizeof(T));
            in.read((char *)&pivot_r, sizeof(T));
            range_mp_[cur] = range(pivot_l, pivot_r);
            to_update.push(cur->get_left());
            to_update.push(cur->get_right());
        }
    }
}

template<class Label, class T>
void KDVirtualSpillTree<Label, T>::save(ofstream & out) const
{
    LOG_INFO("Saving KDTreeNode\n"); 
    this->KDTree<Label, T>::save(out);
    queue<KDTreeNode<Label, T> *> to_save;
    to_save.push((this->get_root()));
    while (!to_save.empty())
    {
        KDTreeNode<Label, T> * cur = to_save.front();
        to_save.pop();
        bool exists = cur != NULL;
        if (exists)
        {
            range cur_range = range_mp_.at(cur);
            out.write((char *)&cur_range.first, sizeof(T));
            out.write((char *)&cur_range.second, sizeof(T));
            to_save.push(cur->get_left());
            to_save.push(cur->get_right());
        }
    }
}

template<class Label, class T>
vector<size_t> KDVirtualSpillTree<Label, T>::subdomain(vector<T> * query, size_t l_c, size_t * number_of_leaves)
{
    LOG_INFO("Enter subdomain\n");
    LOG_FINE("with lc = %ld\n", l_c);
    queue<KDTreeNode<Label, T> *> to_explore;
    set<size_t> domain_st;
    to_explore.push(this->get_root());
    //size_t domain_sum = 0;
    while (!to_explore.empty())
    {
        KDTreeNode<Label, T> * cur = to_explore.front();
        to_explore.pop();
        bool exists = cur != NULL;
        if (exists)
        {
            //size_t tmp_sum = domain_sum + cur->get_domain().size();
            if ((cur->get_left() || cur->get_right()) &&
                cur->get_domain().size() >= l_c)
            {
                range cur_range = range_mp_.at(cur);
                if (cur_range.first <= (*query)[cur->get_index()] &&
                        (*query)[cur->get_index()] < cur_range.second)
                {
                    to_explore.push(cur->get_right());
                    to_explore.push(cur->get_left());
                }
                else if ((*query)[cur->get_index()] <= cur->get_pivot())
                    to_explore.push(cur->get_left());
                else
                    to_explore.push(cur->get_right());
            }
            else
            {
                //domain_sum += cur->get_domain().size();
                (*number_of_leaves)++;
                vector<size_t> l_domain = cur->get_domain();
                for (size_t i = 0; i < l_domain.size(); i++)
                {
                    domain_st.insert(l_domain[i]);
                }
            }
        }
    }
    //LOG_FINE("Exit with domain sum = %ld\n", domain_sum);
    set<size_t>::iterator st_i;
    vector<size_t> domain;
    for (st_i = domain_st.begin(); st_i != domain_st.end(); st_i++)
    {
        domain.push_back(*st_i);
    }
    LOG_INFO("Exit subdomain\n");
    return domain;
}

#endif
