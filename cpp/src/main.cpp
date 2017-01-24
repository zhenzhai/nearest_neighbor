#include <iostream>
#include <cstdio>
#include "test.h"
using namespace std;

typedef unsigned char byte;

/* Please select the dataset you want to run */
const string set_DIR = "mnist";
//const string set_DIR = "cifar";
//const string set_DIR = "songs";
//const string set_DIR = "big5";
//const string set_DIR = "w2v";


int main()
{
    /* For approximate NN search */
    //Test<byte,float> mTest(DIR + set_DIR, 1.4);
    
    /* For exact NN search */
    Test<byte,float> mTest(set_DIR);
    
    /*
     Make sure you set the DIR and set_DIR correctly before you generate data. */

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

