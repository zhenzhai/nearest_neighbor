# Data files output here
* CIFAR1000.mat is the 1000 features extracted from CIFAR. Please see [here](http://ai.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf) for more detail on extracting features from CIFAR. We used K-mean clustering with "triangle" activation function.

* Write train and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ","

* Write train and test labels into files "train_labels" and "test_labels", each line should only have one label.

* Uncomment the following line in main.cpp

		cifar_generate();

* Run makefile