* Download data files from [here](http://yann.lecun.com/exdb/mnist/)

* Place 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', and 't10k-labels-idx1-ubyte' in "raw_data" directory.

* Run
		cd raw_data
		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "raw_data" directory.

* You can now run main.cpp in the src directory

		cd ../..
		make
		./main.o mnist convert 60000 10000 784

* Converted data will be saved in current directory.
