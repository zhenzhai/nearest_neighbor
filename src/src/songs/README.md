# Data files output here
* Download data files from [here](http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset)

* Place all song folders in folder /preprocess.


* Run

		cd preprocess
		python move_files.py
The pythno script will move all the songs files into folder /preprocess.

* Run

		python get_timbre.py

The python script will write train and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, write train and test labels into files "train_labels" and "test_labels", each line with only one label.

* Uncomment the following line in main.cpp

		songs_generate();

* Run makefile