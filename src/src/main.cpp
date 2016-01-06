#include <iostream>
#include <cstdio>
#include "vector_math.h"
#include "nn.h"
#include "kd_tree.h"
#include "n_spill_tree.h"
#include "kd_spill_tree.h"
#include "kd_virtual_spill_tree.h"
#include "data_set.h"
#include "test.h"
#include "cifar.h"
#include "personality.h"
#include "songs.h"
using namespace std;

typedef unsigned char byte;

const string DIR = "/Users/janetzhai/Desktop/nn-xcode/nn-xcode";
//const stirng DIR = "/Users/zhen/Desktop/nn-code";
const string set_DIR = "/personality";

int main() 
{
    //cifar_generate();
    //personality_generate();
    //songs_generate();
    Test<byte,byte> mTest(DIR + "/mnist");
    //Test<byte,float> mTest(DIR + "/cifar");
    //Test<byte,float> mTest(DIR + "/personality");
    //Test<byte,float> mTest(DIR + "/songs");
    mTest.generate_kd_trees();
    mTest.generate_kd_tree_data(DIR + set_DIR);
    //mTest.generate_n_spill_trees();
    //mTest.generate_n_spill_tree_data(DIR + set_DIR);
    //mTest.generate_multi_kd_trees();
    //mTest.generate_multi_kd_tree_data(DIR + set_DIR);
    //mTest.generate_kd_spill_trees();
    //mTest.generate_kd_spill_tree_data(DIR + set_DIR);
    //mTest.generate_pca_trees();
    //mTest.generate_pca_tree_data(DIR + set_DIR);
    //mTest.generate_pca_spill_trees();
    //mTest.generate_pca_spill_tree_data(DIR + set_DIR);
    //mTest.generate_kd_v_spill_trees();
    //mTest.generate_kd_v_spill_tree_data(DIR + set_DIR);
}