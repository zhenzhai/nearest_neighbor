* Download data files from [here](https://www.opensciencedatacloud.org/publicdata/million-song-dataset/).

More information on the Million Songs dataset is [here](http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset).

* The full dataset contain 26 .tar.gz files. Unzip all 26 files and placed in folder "raw_data/raw_songs".

* Run

		cd raw_data
		python preprocess.py

The python script will extract timbre feature from all the songs files and split the dataset into train and test.

The python script will write train and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write dummy train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be in "raw_data" directory.

* You can now run main.cpp in the src directory

		cd ../..
		make
		./main.o songs convert 951175 10000 1200

* Converted data will be saved in current directory.
