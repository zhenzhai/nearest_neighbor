import shutil
import os
import hdf5_getters
import numpy as np
import os
import random

def move_files_out():
	""" Move all the files into one single directory for scanning """
	print "Moving all files for preprocessing ..."
	destination = os.getcwd()
	destination += '/all_song_files'
	walker = os.walk(destination+'/fullset')

	for data in walker:
		for files in data[2]:
			try:
				shutil.move(data[0]+'/'+files, destination)
				print files
			except shutil.Error:
				print files, 'fail'
				continue

	print 'Done moving files'

def get_timbre(file_name):
	h5 = hdf5_getters.open_h5_file_read(file_name)
	timbre = hdf5_getters.get_segments_timbre(h5)
	h5.close()
	return [timbre[:100].flatten()]

def main():
	move_files_out()

	timbre = []
	count = 0
	print "Getting all files ..."
	for filename in os.listdir(os.getcwd()+'/all_song_files'):
		if filename[0] == 'T' and filename[-1] == '5':
			timbre += get_timbre(filename)
	
	# Split files into train and test
	random.seed(1)
	random.shuffle(timbre)
	with open('../raw_data/test_vectors', 'w') as test:
		test.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[:165000])
	with open('../raw_data/train_vectors', 'w') as train:
		train.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[165000:])

	test_labels = np.zeros(shape=(165000,), dtype=np.int)
	train_labels = np.zeros(shape=(835000,), dtype=np.int)
	np.savetxt('../raw_data/test_labels', test_labels, fmt='%d')
	np.savetxt('../raw_data/train_labels', train_labels, fmt='%d')

main()