* Download data files from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

You can find more information about word2vec [here](https://code.google.com/archive/p/word2vec/), the dataset we used is the Google News dataset.

* Place downloaded file in '/preprocess' directory.

* Run distance.c to get raw data files

		cd preprocess
		gcc distance.c -o distance
		./distance .

Two raw datafiles 'GoogleNews' and 'GoogleNewsLabels' will be saved in '/preprocess'.

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "raw_data" directory.

* You can now run main.cpp in the src directory

		cd ..
		make
		./main.o big5 convert 100000 10000 300

* Converted data will be saved in current directory.
