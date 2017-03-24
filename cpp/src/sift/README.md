* Download data files from [here](ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz)

Detail about the dataset 'ANN_SIFT1M' is [here](http://corpus-texmex.irisa.fr/).

* Unzip downloaded files in folder "raw_data".

* Use Matlab run the following in folder "raw_data":
		
		base = fvecs_reader('sift_base.fvecs');
		base = base.';
		query = fvecs_reader('sift_query.fvecs');
		query = query.';
		dlmwrite('train_vec', base);
		dlmwrite('test_vec', query);

* Run the preprocess.py in "raw_data. It will generate dumpy label files for train and test.

		python preprocess.py

* The python script will write train and test feature vectors into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ",". Also, it will write train and test labels into files "train_labels" and "test_labels", each line with only one label. Output files will be saved in "raw_data" directory.

* You can now run main.cpp in the src directory

		cd ..
		make
		./main.o sift convert 985462 10000 128

* Converted data will be saved in current directory.
