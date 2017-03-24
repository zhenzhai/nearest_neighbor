from collections import defaultdict
import numpy as np
import random

def remove_duplicate(train_data):
    print 'remove duplicate from data of size ', len(train_data)
    sift_vec = []
    sift_lab = []
    unique_dict = defaultdict(int)

    index = 0
    count = 0
    for line in train_data:
        string_data = str(line)
        if not unique_dict[string_data]:
            unique_dict[string_data] = 1
            sift_vec.append(line)
            sift_lab.append(int(0))
            count += 1
        index += 1
        if index % 100000 == 0:
            print '    at index ', index
    print 'Done'
    print 'Down size from ', index, ' to ', count
    return sift_vec, sift_lab


def write_train_test(train, train_lab, test):
    sift_train_vec = train
    sift_train_lab = train_lab
    sift_test_vec = test
    sift_test_lab = []
    for i in xrange(len(test)):
        sift_test_lab.append(int(0))
        if i % 10000 == 0:
            print "    at index", i
    sift_test_lab = np.array(sift_test_lab)
    print "Done"
    print "train vec of size: ", len(sift_train_vec)
    print "test vec of size: ", len(sift_test_vec)
    print "train lab of size: ", len(sift_train_lab)
    print "test lab of size: ", len(sift_test_lab)

    print "write to files ..."
    print "writing train vector"
    train_vec = open("train_vectors", "w")
    np.savetxt(train_vec, sift_train_vec, delimiter=',', fmt='%i')
    print "writing train label ..."
    train_lab = open("train_labels", "w")
    np.savetxt(train_lab, sift_train_lab, delimiter=',', fmt='%i')
    print "writing test vector ..."
    test_vec = open("test_vectors", "w")
    np.savetxt(test_vec, sift_test_vec, delimiter=',', fmt='%i')
    print "writing test label ..."
    test_lab = open("test_labels", "w")
    np.savetxt(test_lab, sift_test_lab, delimiter=',', fmt='%i')


if __name__ == '__main__':
    print 'Reading raw data file ...'
    train_file = open('train_vec', 'r')
    train = np.loadtxt(train_file, delimiter = ",")
    test_file = open('test_vec', 'r')
    test = np.loadtxt(test_file, delimiter = ",")

    vec, lab = remove_duplicate(train)
    write_train_test(vec, lab, test)