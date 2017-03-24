* Download data files from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

You can find more information about word2vec [here](https://code.google.com/archive/p/word2vec/), the dataset we used is the Google News dataset.

* Place downloaded file in '/raw_data' directory.

* Run distance.c to get raw data files

		cd raw_data
		gcc distance.c -o distance
		./distance .

Two raw datafiles 'GoogleNewsData' and 'GoogleNewsLabels' will be saved in '/raw_data'.

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one integer label. Output files will be in "raw_data" directory.

Word2Vec data has labels as words in string. We create a mapping from each string label to a integer. This mapping will be a python dictionary saved in pickle file "label_mapping.pkl".

* You can now run main.cpp in the src directory

		cd ../..
		make
		./main.o w2v convert 2990000 10000 300

* Converted data will be saved in current directory.
