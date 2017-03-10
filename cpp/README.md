# How to run our project

* All source files are located in 'src' folder.

* Before running the 'src/main.cpp', please take a look at each of the dataset folders. We have [MNIST](https://github.com/zhenzhai/nearest_neighbor/tree/master/cpp/src/mnist), [CIFAR](https://github.com/zhenzhai/nearest_neighbor/tree/master/cpp/src/cifar), [Word2Vec](https://github.com/zhenzhai/nearest_neighbor/tree/master/cpp/src/w2v), [BIG5](https://github.com/zhenzhai/nearest_neighbor/tree/master/cpp/src/big5), and [Million Songs](https://github.com/zhenzhai/nearest_neighbor/tree/master/cpp/src/songs).

* You will first need to download the corresponding dataset and convert the raw data. Please see README in each of the dataset folders to find instructions.

* Then you can select which of these dataset you want to run. You can only run one dataset at a time.

* We explored 7 different data structures, they are k-d trees, randomized k-d trees, RP trees, V^2 trees, PCA trees, spill trees, and virtual spill trees.

* Please edit the 'src/main.cpp' correspondingly and choose the data structures you want to run.
