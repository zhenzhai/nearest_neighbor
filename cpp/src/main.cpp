#include <iostream>
#include <sstream>
#include <cstdio>
#include "test.h"
#include "data_convert.h"
using namespace std;

typedef unsigned char byte;

/* Please select the dataset you want to run */
//const string set_DIR = "mnist";
//const string set_DIR = "cifar";
//const string set_DIR = "songs";
//const string set_DIR = "big5";
//const string set_DIR = "w2v";


int main(int argc, char* argv[])
{
    if (argc < 2 || (argc > 2 && argc < 6)) {
        cerr << "Usage: " << endl;
        cerr << "   1. Convert Data "<< argv[0] << " DataName(mnist/cifar/songs/big5/w2v/sift) convert train_size test_size width" << endl;
        cerr << "   2. Run Trees " << argv[0] << " DataName(mnist, cifar, songs, big5, w2v, sift)" << endl;
        cerr << "   3. Run Specific Tree " << argv[0] << " DataName(mnist/cifar/songs/big5/w2v/sift) tree_name(kd/rkd/rp/pca/pca_spill/kd_spill/kd_v_spill)" << endl;
        return 1;
    }
    string set_DIR = argv[1];
    if (argc == 6) {
        size_t train_size = atoi(argv[3]);
        size_t test_size = atoi(argv[4]);
        size_t width = atoi(argv[5]);
        cout << "Start to convert data " << set_DIR << endl;
        data_generate(set_DIR, train_size, test_size, width);
    } else if (argc == 3) {
        string tree = argv[2];
        cout << "Start to build " << tree << endl;
        /* For approximate NN search */
        //Test<byte,float> mTest(DIR + set_DIR, 1.4);
        
        /* For exact NN search */
        Test<byte,float> mTest(set_DIR);
        
        if (tree == "kd") {
            mTest.generate_kd_trees();
            mTest.generate_kd_tree_data(set_DIR);
        } else if (tree == "rkd") {
            mTest.generate_rkd_trees();
            mTest.generate_rkd_tree_data(set_DIR);
        } else if (tree == "rp") {
            mTest.generate_rp_trees();
            mTest.generate_rp_tree_data(set_DIR);
        } else if (tree == "pca") {
            mTest.generate_pca_trees();
            mTest.generate_pca_tree_data(set_DIR);
        } else if (tree == "pca_spill") {
            mTest.generate_pca_spill_trees();
            mTest.generate_pca_spill_tree_data(set_DIR);
        } else if (tree == "kd_spill") {
            mTest.generate_kd_spill_trees();
            mTest.generate_kd_spill_tree_data(set_DIR);
        } else if (tree == "kd_v_spill") {
            mTest.generate_kd_v_spill_trees();
            mTest.generate_kd_v_spill_tree_data(set_DIR);
        }

    } else {
        /* For approximate NN search */
        //Test<byte,float> mTest(DIR + set_DIR, 1.4);
    
        /* For exact NN search */
        Test<byte,float> mTest(set_DIR);
        
        mTest.generate_kd_trees();
        mTest.generate_kd_tree_data(set_DIR);
    
        mTest.generate_rkd_trees();
        mTest.generate_rkd_tree_data(set_DIR);
    
        mTest.generate_rp_trees();
        mTest.generate_rp_tree_data(set_DIR);
    
        mTest.generate_rp_select_trees();
        mTest.generate_rp_select_tree_data(set_DIR);
    
        mTest.generate_pca_trees();
        mTest.generate_pca_tree_data(set_DIR);
    
        mTest.generate_pca_spill_trees();
        mTest.generate_pca_spill_tree_data(set_DIR);
    
        mTest.generate_kd_spill_trees();
        mTest.generate_kd_spill_tree_data(set_DIR);
    
        mTest.generate_kd_v_spill_trees();
        mTest.generate_kd_v_spill_tree_data(set_DIR);
    }
}

