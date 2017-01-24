* Download data files from [here](http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset)

* Place downloaded folder "fullset" in folder "/preprocess".

* Run

		cd preprocess
		python preprocess.py

The python script will move all the songs files into folder /all_song_files and split the timbre features into train and test.

The python script will write train and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "convert_data" directory.

* Run to convert data

		cd convert_data
		make
		./main.o

* Converted data will be saved in current directory.
