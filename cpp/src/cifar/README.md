* Download data file [here](https://drive.google.com/file/d/0B7T8AxpaeM47M1pLNnk5RktsbHc/view?usp=sharing).

We extracted 1000 features from CIFAR using [Honglak Lee](http://web.eecs.umich.edu/~honglak/)'s K-means feature learning code. Please see [publication](http://web.eecs.umich.edu/~honglak/aistats11-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf) and [code](http://cs.stanford.edu/~acoates/papers/kmeans_demo.tgz).

* Unzip the downloaded folder and put all files from folder into current directory.

* Run to convert data

		cd convert_data
		make
		./main.o

* Converted data will be saved in current directory.
