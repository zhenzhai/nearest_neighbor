* Download data files from [here](http://mypersonality.org/wiki/doku.php?id=download_databases)

We used 'Psychological Profiles' -> 'BIG5 Personality Scores' -> 'Item-level data for a 336-item questionnaire'.

You will need to register as a collaborator before you download the dataset.

Detail about the dataset is [here](http://mypersonality.org/wiki/doku.php?id=list_of_variables_available#personality_scores).

* Place downloaded csv file in current directory.

* Run

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be saved in "raw_data" directory.

* You can now run main.cpp in the src directory

		cd ..
		make
		./main.o big5 convert 100000 10000 100

* Converted data will be saved in current directory.
