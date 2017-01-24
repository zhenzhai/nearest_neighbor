from collections import defaultdict

def remove_duplicate():
    print "Start to remove duplicate data ..."
    unique_dict = defaultdict(int)
    index_array = []
    new_data = open("simple_data", "w")
    new_label = open("simple_label", "w")

    f = open('GoogleNews', 'r')
    lines_v = f.readlines()
    f.close()
    fl = open('GoogleNewsLabels', 'r')
    lines_l = fl.readlines()
    fl.close()

    index = 0
    count = 0
    print "Loop for data of size ", len(lines_v)
    for l in xrange(len(lines_v)):
        line = lines_v[l]
        data = line.strip().split(',')
        string_data = str(data)
        if not unique_dict[string_data]:
            unique_dict[string_data] = 1
            new_data.write(lines_v[l])
            new_label.write(lines_l[l])
            count += 1
        index += 1
        if index % 100000 == 0:
            print "    at index ", index
    new_data.close()
    new_label.close()
    print "Done."
    print "Down size dataset from size ", index, ' to ', count
    print "Simplified data saved in 'simple_data' and 'simple_label' files."

import random

def sampling_data(train_size, test_size):
    print 'Start sampling ...'
    print 'Reading simple_data and simple_label'
    f = open("simple_data", 'r')
    lines_v = f.readlines()
    fl = open("simple_label", 'r')
    lines_l= fl.readlines()

    print 'sample from dataset of size ', len(lines_v)
    print 'sample ', train_size, ' to be train'
    print 'sample ', test_size, ' to be test'

    random.seed(1)
    sam = random.sample(range(0,len(lines_v)),train_size+test_size)
    sam = sorted(sam)
    random.seed(1)
    test_sam = sorted(random.sample(sam, test_size))

    train_vec = []
    train_lab = []
    test_vec = []
    test_lab = []

    for i in xrange(len(lines_v)):
        if len(sam) == 0:
            break
        v = lines_v[i]
        l = lines_l[i]
        if i == sam[0]:
            if len(test_sam) == 0:
                train_vec.append(v)
                train_lab.append(l)
            elif i != test_sam[0]:
                train_vec.append(v)
                train_lab.append(l)
            else:
                test_vec.append(v)
                test_lab.append(l)
                test_sam = test_sam[1:]
            sam = sam[1:]
        if i % 100000 == 0:
            print "    at index", i
    print "Done"
    print "train vec of size: ", len(train_vec)
    print "test vec of size: ", len(test_vec)

    print "Writing to files"
    train_vec = open("convert_data/train_vectors", "w")
    for b in train_vec:
        train_vec.write(','.join(b) + '\n')
    print "writing train vector"
    train_lab = open("convert_data/train_labels", "w")
    for b in train_lab:
        train_lab.write(b + '\n')
    print "writing train label"
    test_vec = open("convert_data/test_vectors", "w")
    for b in test_vec:
        test_vec.write(','.join(b) + '\n')
    print "writing test vector"
    test_lab = open("convert_data/test_labels", "w")
    for b in test_lab:
        test_lab.write(b + '\n')
    print "writing test label"

if __name__ == "__main__":
    remove_duplicate()
    sampling_data(100000, 10000)
