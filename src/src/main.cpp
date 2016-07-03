#include <iostream>
#include <cstdio>
#include "test.h"
#include "mnist.h"
#include "cifar.h"
#include "personality.h"
#include "songs.h"
using namespace std;

typedef unsigned char byte;

/* Please set the following variable before run */
const string DIR = "nearest_neighbor/src/src";

//const string set_DIR = "/mnist";
//const string set_DIR = "/cifar";
//const string set_DIR = "/personality";
//const string set_DIR = "/songs";

int main()
{
    /* The following functions are used to generate data
     They only need to be run once for each data set*/
    
    //mnist_generate();
    //cifar_generate();
    //personality_generate();
    //songs_generate();
    
    Test<byte,float> mTest(DIR + set_DIR);
    
    /* Please uncomment the corresponding algorithm below.
     Make sure you set the DIR and set_DIR correctly before you generate data. */

    //mTest.generate_kd_trees();
    //mTest.generate_kd_tree_data(DIR + set_DIR);
    //mTest.generate_n_spill_trees();
    //mTest.generate_n_spill_tree_data(DIR + set_DIR);
    //mTest.generate_rp_trees();
    //mTest.generate_rp_tree_data(DIR + set_DIR);
    //mTest.generate_kd_spill_trees();
    //mTest.generate_kd_spill_tree_data(DIR + set_DIR);
    //mTest.generate_pca_trees();
    //mTest.generate_pca_tree_data(DIR + set_DIR);
    //mTest.generate_pca_spill_trees();
    //mTest.generate_pca_spill_tree_data(DIR + set_DIR);
    //mTest.generate_kd_v_spill_trees();
    //mTest.generate_kd_v_spill_tree_data(DIR + set_DIR);
}

