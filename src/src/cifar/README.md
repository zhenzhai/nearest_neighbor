# Data files output here
* Download data file [here](https://www.dropbox.com/s/n0fbjqiatblfja4/cifar.zip?dl=0). We extract 1000 features from CIFAR. Please see [here](http://ai.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf) for more detail. We used K-mean clustering with "triangle" activation function.

* Unzip the downloaded folder and put all files from folder into here.

* Uncomment the following line in main.cpp

		cifar_generate();

* Run makefile