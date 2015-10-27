/* 
 * File             : test.h
 * Date             : 2014-5-29
 * Summary          : Provides the testing interface for the data structures
 */
#ifndef TEST_H_
#define TEST_H_

#include <map>
#include <thread>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include "logging.h"
#include "data_set.h"
#include "kd_tree.h"
#include "n_spill_tree.h"
#include "multi_kd_tree.h"
#include "kd_spill_tree.h"
#include "kd_virtual_spill_tree.h"
#include "pca_tree.h"
#include "pca_spill_tree.h"
using namespace std;

#ifndef NN_DATA_TYPES_
#define NN_DATA_TYPES_
#define COL_W           (20)
#define ERROR_RATE      (0x0001)
#define TRUE_NN         (0x0002)
#define SUBDOMAIN       (0x0004)
#endif

static double multiple_tree     = 4;
static double min_leaf  = 0.005;//0.0001;
static double l []      = {0.015, 0.03, 0.06, 0.09, 0.1, 0.13, 0.15, 0.17, 0.19, 0.21};//{0.105, 0.12, 0.135, 0.15};
const size_t l_len      = 10;
static double a []      = {0.05, 0.1};//{0.07, 0.1, 0.12, 0.15, 0.17, 0.19, 0.21, 0.23};//, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39};
const size_t a_len      = 2;//9;

template<class Label, class T>
class Test
{
protected:
    const string base_dir_;
    DataSet<Label, T> * trn_st_;
    DataSet<Label, T> * tst_st_;
    map<vector<T> *, vector<size_t>> nn_mp_;
public:
    Test(string base_dir);
    ~Test();

    void s_kd_tree(double ll) {
        stringstream dir; 
        dir << base_dir_ << "/kd_tree_" << setprecision(2) << ll;
        KDTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }

    void generate_kd_trees() {
        s_kd_tree(min_leaf);
    }
    
    void s_n_spill_tree(double ll, double la) {
        stringstream dir;
        dir << base_dir_ << "/n_spill_tree_" << setprecision(2) << la << "_" << ll;
        NSpillTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }
    
    void generate_n_spill_trees() {
        thread t [a_len];
        for (size_t i = 0; i < a_len; i++) {
            t[i] = thread(&Test<Label, T>::s_n_spill_tree, this, min_leaf, a[i]);
        }
        for (size_t i = 0; i < a_len; i++) {
            t[i].join();
        }
    }
    
    void s_multi_kd_tree(double ll, int n) {
        stringstream dir;
        for (int i=5; i<=n; i++) {
            dir << base_dir_ << "/multi_kd_tree" << i << "_" << setprecision(2) << ll;
            MultiKDTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), *trn_st_);
            ofstream tree_out (dir.str());
            tree.save(tree_out);
            tree_out.close();
            dir.str("");
        }
    }
    
    void generate_multi_kd_trees() {
        s_multi_kd_tree(min_leaf, multiple_tree);
    }

    void s_kd_spill_tree(double ll, double la) {
        stringstream dir; 
        dir << base_dir_ << "/kd_spill_tree_" << setprecision(3) << la << "_" << ll;
        KDSpillTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), la, *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }

    void generate_kd_spill_trees() {
        thread t [a_len];
        for (size_t i = 0; i < a_len; i++) {
            t[i] = thread(&Test<Label, T>::s_kd_spill_tree, this, min_leaf, a[i]);
        }
        for (size_t i = 0; i < a_len; i++) {
            t[i].join();
        }
    }

    void s_kd_v_spill_tree(double ll, double la)
    {
        stringstream dir; 
        dir << base_dir_ << "/kd_v_spill_tree_" << setprecision(2) << la << "_" << ll;
        KDVirtualSpillTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), la, *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }

    void generate_kd_v_spill_trees()
    {
        thread t [a_len];
        for (size_t i = 0; i < a_len; i++) {
            t[i] = thread(&Test<Label, T>::s_kd_v_spill_tree, this, min_leaf, a[i]);
        }
        for (size_t i = 0; i < a_len; i++) {
            t[i].join();
        }
    }

    void s_pca_tree(double ll)
    {
        stringstream dir; 
        dir << base_dir_ << "/pca_tree_" << setprecision(2) << ll;
        PCATree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }

    void generate_pca_trees()
    {
        s_pca_tree(min_leaf);
    }

    void s_pca_spill_tree(double ll, double la)
    {
        stringstream dir; 
        dir << base_dir_ << "/pca_spill_tree_" << setprecision(2) << la << "_" << ll;
        PCASpillTree<Label, T> tree ((size_t)(ll * (*trn_st_).size()), la, *trn_st_);
        ofstream tree_out (dir.str());
        tree.save(tree_out);
        tree_out.close();
    }

    void generate_pca_spill_trees()
    {
        thread t [a_len];
        for (size_t i = 0; i < a_len; i++) {
            t[i] = thread(&Test<Label, T>::s_pca_spill_tree, this, min_leaf, a[i]);
        }
        for (size_t i = 0; i < a_len; i++) {
            t[i].join();
        }
    }

    void s_kd_tree_data(double ll, string * result)
    {
        stringstream dir; 
        dir << base_dir_ << "/kd_tree_" << setprecision(2) << min_leaf;
        ifstream tree_in (dir.str());
        KDTree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size())));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i], subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data << endl;
        *result = data.str();
    }

    void generate_kd_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/n_spill_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out << endl;
        thread t [l_len];
        string r [l_len];
        for (size_t i = 0; i < l_len; i++) {
            t[i] = thread(&Test::s_kd_tree_data, this, l[i], &(r[i]));
        }
        for (size_t i = 0; i < l_len; i++) {
            t[i].join();
            dat_out << r[i];
        }
        dat_out.close();
    }
    
    void s_n_spill_tree_data(double ll, double la, string * result)
    {
        stringstream dir;
        dir << base_dir_ << "/n_spill_tree_" << setprecision(2) << min_leaf;
        ifstream tree_in (dir.str());
        NSpillTree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size())));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i], subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) <<  la;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data << endl;
        *result = data.str();
    }
    
    void generate_n_spill_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/n_spill_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "alpha";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out <<  setw(COL_W) << "space blowup";
        dat_out << endl;
        thread t [l_len][a_len];
        string r [l_len][a_len];
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j] = thread(&Test<Label, T>::s_n_spill_tree_data, this, l[i], a[j], &(r[i][j]));
            }
        }
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j].join();
                dat_out << r[i][j];
            }
        }
        dat_out.close();
    }
    
    void s_multi_kd_tree_data(double ll, string * result, int n)
    {
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            stringstream dir;
            vector<size_t> nn_domain;
            for (int j=1; j<=n; j++) {
                dir << base_dir_ << "/multi_kd_tree" << j << "_" << setprecision(2) << min_leaf;
                string test = dir.str();
                ifstream tree_in (dir.str());
                MultiKDTree<Label, T> tree (tree_in, *trn_st_);
                DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)((ll / n) * (*trn_st_).size())));
                if (nn_domain.size() == 0){
                    nn_domain = subSet.get_domain();
                } else {
                    vector<size_t> d = subSet.get_domain();
                    nn_domain.insert(nn_domain.end(), d.begin(), d.end());
                }
                subdomain_count += subSet.size();
                dir.str("");
            }
            vector<T> * nn = nearest_neighbor((*tst_st_)[i], (*trn_st_).subset(nn_domain));
            Label nn_lbl = (*trn_st_).get_label(nn);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
        }
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data << endl;
        *result = data.str();
    }
    
    void generate_multi_kd_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/4multi_kd_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out << endl;
        thread t [l_len];
        string r [l_len];
        for (size_t i = 0; i < l_len; i++) {
            t[i] = thread(&Test::s_multi_kd_tree_data, this, l[i], &(r[i]), multiple_tree);
        }
        for (size_t i = 0; i < l_len; i++) {
            t[i].join();
            dat_out << r[i];
        }
        dat_out.close();
    }

    void s_kd_spill_tree_data(double ll, double la, string * result)
    {
        stringstream dir; 
        dir << base_dir_ << "/kd_spill_tree_" << setprecision(2) << la << "_" << min_leaf;
        ifstream tree_in (dir.str());
        KDSpillTree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size())));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i],subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        
        //calculate space blowup
        size_t space_blowup = 0;
        int tree_height = 0;
        int number_leaves = 1;
        LOG_INFO("Space calculation\n");
        size_t l_c = (size_t)(ll * (*trn_st_).size());
        queue<KDTreeNode<Label, T> *> expl;
        expl.push(tree.get_root());
        while (!expl.empty())
        {
            KDTreeNode<Label, T> * cur = expl.front();
            expl.pop();
            if (cur->get_left() && cur->get_right() && cur->get_domain().size() >= l_c) {
                expl.push(cur->get_right());
                size_t dsize = cur->get_domain().size();
                space_blowup = dsize * number_leaves;
                tree_height++;
                number_leaves = number_leaves * 2;
            }
            else
                break;
        }
        LOG_INFO("Done space calculation\n");
        size_t root_size = tree.get_root()->get_domain().size();
        
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) <<  la;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << 1. * space_blowup/root_size;
        data << endl;
        *result = data.str();
    }

    void generate_kd_spill_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/kd_spill_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "alpha";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out <<  setw(COL_W) << "space blowup";
        dat_out << endl;
        thread t [l_len][a_len];
        string r [l_len][a_len];
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j] = thread(&Test<Label, T>::s_kd_spill_tree_data, this, l[i], a[j], &(r[i][j]));
            }
        }
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j].join();
                dat_out << r[i][j];
            }
        }
        dat_out.close();
    }

    void s_kd_v_spill_tree_data(double ll, double la, string * result)
    {
        stringstream dir; 
        dir << base_dir_ << "/kd_v_spill_tree_" << setprecision(2) << la << "_" << min_leaf;
        ifstream tree_in (dir.str());
        KDVirtualSpillTree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        size_t number_of_leaves = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size()), & number_of_leaves));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i], subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) <<  la;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (number_of_leaves * 1. / (*tst_st_).size());
        data << endl;
        *result = data.str();
    }

    void generate_kd_v_spill_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/kd_v_spill_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "alpha";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out <<  setw(COL_W) << "number of leaves";
        dat_out << endl;
        thread t [l_len][a_len];
        string r [l_len][a_len];
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j] = thread(&Test<Label, T>::s_kd_v_spill_tree_data, this, l[i], a[j], &(r[i][j]));
            }
        }
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j].join();
                dat_out << r[i][j];
            }
        }
        dat_out.close();
    }

    void s_pca_tree_data(double ll, string * result)
    {
        stringstream dir; 
        dir << base_dir_ << "/pca_tree_" << setprecision(2) << min_leaf;
        ifstream tree_in (dir.str());
        PCATree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size())));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i], subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data << endl;
        *result = data.str();
    }

    void generate_pca_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/pca_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out << endl;
        thread t [l_len];
        string r [l_len];
        for (size_t i = 0; i < l_len; i++) {
            t[i] = thread(&Test::s_pca_tree_data, this, l[i], &(r[i]));
        }
        for (size_t i = 0; i < l_len; i++) {
            t[i].join();
            dat_out << r[i];
        }
        dat_out.close();
    }

    void s_pca_spill_tree_data(double ll, double la, string * result)
    {
        stringstream dir; 
        dir << base_dir_ << "/pca_spill_tree_" << setprecision(2) << la << "_" << min_leaf;
        ifstream tree_in (dir.str());
        PCASpillTree<Label, T> tree (tree_in, *trn_st_);
        size_t error_count = 0;
        size_t true_nn_count = 0;
        unsigned long long subdomain_count = 0;
        for (size_t i = 0; i < (*tst_st_).size(); i++) {
            DataSet<Label, T> subSet = (*trn_st_).subset(tree.subdomain((*tst_st_)[i], (size_t)(ll * (*trn_st_).size())));
            vector<T> * nn_vtr = nearest_neighbor((*tst_st_)[i],
                                 subSet);
            Label nn_lbl = (*trn_st_).get_label(nn_vtr);
            if (nn_lbl != (*tst_st_).get_label(i))
                error_count++;
            if (nn_vtr == (*trn_st_)[nn_mp_[(*tst_st_)[i]][0]])
                true_nn_count++;
            subdomain_count += subSet.size();
        }
        
        //calculate space blowup
        size_t space_blowup = 0;
        int tree_height = 0;
        int number_leaves = 1;
        LOG_INFO("Space calculation\n");
        size_t l_c = (size_t)(ll * (*trn_st_).size());
        queue<PCATreeNode<Label, T> *> expl;
        expl.push(tree.get_root());
        while (!expl.empty())
        {
            PCATreeNode<Label, T> * cur = expl.front();
            expl.pop();
            if (cur->get_left() && cur->get_right() && cur->get_domain().size() >= l_c) {
                expl.push(cur->get_right());
                size_t dsize = cur->get_domain().size();
                space_blowup = dsize * number_leaves;
                tree_height++;
                number_leaves = number_leaves * 2;
            }
            else
                break;
        }
        LOG_INFO("Done space calculation\n");
        size_t root_size = tree.get_root()->get_domain().size();

        stringstream data;
        data <<  setw(COL_W) <<  ll;
        data <<  setw(COL_W) <<  la;
        data <<  setw(COL_W) << (error_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (true_nn_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << (subdomain_count * 1. / (*tst_st_).size());
        data <<  setw(COL_W) << 1. * space_blowup/root_size;
        data << endl;
        *result = data.str();
    }

    void generate_pca_spill_tree_data(string out_dir)
    {
        ofstream dat_out (out_dir + "/pca_spill_tree.dat");
        dat_out <<  setw(COL_W) << "leaf";
        dat_out <<  setw(COL_W) << "alpha";
        dat_out <<  setw(COL_W) << "error rate";
        dat_out <<  setw(COL_W) << "true nn";
        dat_out <<  setw(COL_W) << "subdomain";
        dat_out <<  setw(COL_W) << "space blowup";
        dat_out << endl;
        thread t [l_len][a_len];
        string r [l_len][a_len];
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j] = thread(&Test<Label, T>::s_pca_spill_tree_data, this, l[i], a[j], &(r[i][j]));
            }
        }
        for (size_t i = 0; i < l_len; i++) {
            for (size_t j = 0; j < a_len; j++) {
                t[i][j].join();
                dat_out << r[i][j];
            }
        }
        dat_out.close();
    }
};

template<class Label, class T>
Test<Label, T>::Test(string base_dir) :
  base_dir_ (base_dir)
{
    LOG_INFO("Loading Data Sets\n");
    ifstream trn_vtr_in (base_dir + "/trn_vtr", ios::binary);
    ifstream tst_vtr_in (base_dir + "/tst_vtr", ios::binary);
    trn_st_ = new DataSet<Label, T>(trn_vtr_in);
    tst_st_ = new DataSet<Label, T>(tst_vtr_in);
    trn_vtr_in.close();
    tst_vtr_in.close();
    LOG_INFO("Labeling Data Sets\n");
    ifstream trn_lbl_in (base_dir + "/trn_lbl", ios::binary);
    ifstream tst_lbl_in (base_dir + "/tst_lbl", ios::binary);
    trn_st_->label(trn_lbl_in);
    tst_st_->label(tst_lbl_in);
    trn_lbl_in.close();
    tst_lbl_in.close();
    LOG_INFO("Success!\n");
    ifstream nn_dat_in (base_dir + "/k_true_nn", ios::binary);
    if (nn_dat_in.good()) {
        size_t k;
        nn_dat_in.read((char *)&k, sizeof(size_t));
        LOG_INFO("File \"k_true_nn\" found!!!\n");
        LOG_INFO("Parsing file with k = %ld\n", k);
        for (size_t i = 0; i < tst_st_->size(); i++) {
            for (int j = 0; j < k; j++) {
                size_t nn; 
                nn_dat_in.read((char *)&nn, sizeof(size_t));
                nn_mp_[(*tst_st_)[i]].push_back(nn);
            }
        }
        nn_dat_in.close();
        LOG_INFO("Success!\n");
    } else {
        LOG_WARNING("File \"k_true_nn\" not found!!!\n");
        size_t k = 10;
        nn_dat_in.close();
        LOG_WARNING("Generating \"k_true_nn\" with k = %ld\n", k);
        ofstream nn_dat_out (base_dir + "/k_true_nn", ios::binary);
        nn_dat_out.write((char *)&k, sizeof(size_t));
        for (size_t i = 0; i < tst_st_->size(); i++) {
            DataSet<Label, T> l_st = k_nearest_neighbor(k, (*tst_st_)[i], *trn_st_);
            for (size_t j = 0; j < k; j++) {

                nn_dat_out.write((char *)&(l_st.get_domain()[j]), sizeof(size_t));
                nn_mp_[(*tst_st_)[i]].push_back(l_st.get_domain()[j]);
            }
        }
        nn_dat_out.close();
        LOG_INFO("Success!\n");
    }
}

template<class Label, class T>
Test<Label, T>::~Test()
{
    delete trn_st_;
    delete tst_st_;
}

#endif /* TEST_H_ */
