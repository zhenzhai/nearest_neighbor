* Download data files from [here](http://yann.lecun.com/exdb/mnist/)

* Place 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', and 't10k-labels-idx1-ubyte' in current directory.

* Run

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "convert_data" directory.

* Run to convert data

		cd convert_data
		make
		./main.o

* Converted data will be saved in current directory.
