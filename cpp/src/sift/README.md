* Download data files from [here](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)

Detail about the dataset 'ANN_SIFT1M' is [here](http://corpus-texmex.irisa.fr/).

* Unzip downloaded files in current directory.

* Run

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be saved in "convert_data" directory.

* Run to convert data

		cd convert_data
		make
		./main.o

* Converted data will be saved in current directory.
