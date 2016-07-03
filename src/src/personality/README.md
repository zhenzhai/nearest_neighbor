# Data files output here
* Download data files from [here](http://personality-testing.info/_rawdata/), we used Cattell's 16 Personality Factors Test.

* Remove any entries that are not interger. All feature should be a number between 1 and 5.

* Split the data randomly into training set and test set.


* Write training and test features into files "train_vectors" and "test_vectors" correspondingly, each feature seperated with ","

* Write training and test labels into files "train_labels" and "test_labels", each line should only have one label.

* Uncomment the following line in main.cpp

		personality_generate();

* Run makefile