* Download data files from [here](http://yann.lecun.com/exdb/mnist/)

* Place 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', and 't10k-labels-idx1-ubyte' here.

* Run

		python Preprocess.py

* The python script will write train and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "src" directory.

* Run to convert data

		cd src
		make
		./main.o
