# Data files output here
* Download data file [here](https://www.dropbox.com/s/n0fbjqiatblfja4/cifar.zip?dl=0). We extract 1000 features from CIFAR using [Honglak Lee](http://web.eecs.umich.edu/~honglak/)'s K-means feature learning code. Please see [publication](http://web.eecs.umich.edu/~honglak/aistats11-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf) and [code](http://cs.stanford.edu/~acoates/papers/kmeans_demo.tgz).

* Unzip the downloaded folder and put all files from folder into here.

* Uncomment the following line in main.cpp

		cifar_generate();

* Run makefile