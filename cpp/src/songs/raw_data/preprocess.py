import shutil
import os
import hdf5_getters
import numpy as np
import os
import random
import fnmatch

def get_timbre(file_name):
	h5 = hdf5_getters.open_h5_file_read(file_name)
	timbre = hdf5_getters.get_segments_timbre(h5)
	h5.close()
	return [timbre[:100].flatten()]

def main(test_size):

	print "Getting all files ..."
	all_files = []
	for root, dirnames, filenames in os.walk('raw_songs'):
		for filename in fnmatch.filter(filenames, '*.h5'):
			all_files.append(os.path.join(root, filename))

	print "Extracting timbre feature from ", len(all_files),  " files ..."
	timbre = []
	count = 0
	for filename in all_files:
		if count % 10000 == 0:
			print "		at file count ", count
		timbre += get_timbre(filename)
		count += 1
	
	dim = len(timbre[0])
	print "Extract timbre of dimension ", dim
	# Split files into train and test
	print "Spliting data into train and test ..."
	random.seed(1)
	random.shuffle(timbre)
	print "Writing test data ..."
	with open('test_vectors', 'w') as test:
		test.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[:test_size])
	print "Writing train data ..."
	with open('train_vectors', 'w') as train:
		train.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[test_size:])

	print "Writing dummy label files ..."
	train_size = len(timbre) - test_size
	test_labels = np.zeros(shape=(test_size,), dtype=np.int)
	train_labels = np.zeros(shape=(train_size,), dtype=np.int)
	np.savetxt('test_labels', test_labels, fmt='%d')
	np.savetxt('train_labels', train_labels, fmt='%d')
	print "Done preprocessing train data of size ", train_size, " and test data of size ", test_size, " with dimension ", dim

main(10000)