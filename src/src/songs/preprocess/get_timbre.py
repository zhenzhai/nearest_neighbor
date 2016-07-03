import hdf5_getters
import numpy as np
import os
import random

def get_timbre(file_name):
	h5 = hdf5_getters.open_h5_file_read(file_name)
	timbre = hdf5_getters.get_segments_timbre(h5)
	h5.close()
	return [timbre[:100].flatten()]

def main():
	timbre = []
	count = 0
	for filename in os.listdir(os.getcwd()):
		if filename[0] == 'T' and filename[-1] == '5':
			timbre += get_timbre(filename)
			print filename
	
	random.shuffle(timbre)
	with open('test_vectors', 'w') as test:
		test.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[:165000])
	with open('train_vectors', 'w') as train:
		train.writelines(','.join(str(j) for j in i) + '\n' for i in timbre[165000:])
	test_labels = np.zeros(shape=(165000,), dtype=np.int)
	train_labels = np.zeros(shape=(835000,), dtype=np.int)
	np.savetxt('test_labels', test_labels, fmt='%d')
	np.savetxt('train_labels', train_labels, fmt='%d')

main()

