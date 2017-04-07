from collections import defaultdict
import random
import pickle as pkl
import csv

def remove_duplicate(data100):
    print 'remove duplicate from data of size ', len(data100)
    big5_vec = []
    big5_lab = []
    unique_dict = defaultdict(int)

    index = 0
    count = 0
    for data in data100:
    	line = data[:100]
        string_data = str(line)
        if not unique_dict[string_data]:
            unique_dict[string_data] = 1
            v = []
            for t in line:
            	if t == '':
            		v.append('0')
            	else:
            		v.append(t)
            l = ''.join(data[104:109])
            big5_vec.append(v)
            big5_lab.append(l)
            count += 1
        index += 1
        if index % 100000 == 0:
            print '    at index ', index
            print '			sample print: ', v[:10]
            print '			sample label: ', l
    print 'Done'
    print 'Down size from ', index, ' to ', count
    return big5_vec, big5_lab

def filter_data():
    print 'Reading raw data file ...'
    big5_file = csv.reader(open('big5.csv', "r"))
    big5 = [b[1:] for b in big5_file]

    ### Only capture data with 100 questions
    print 'Filtering raw data ...'
    big5_100data = []
    for b in big5[1:]:
        if b[101] == "100":
            big5_100data.append(b[:109])

    vec, lab = remove_duplicate(big5_100data)
    return vec, lab

def split_train_test(vec, lab, train_size, test_size):
    print 'Start sampling ...'
    print 'sample from data size ', len(vec)
    print 'sample ', train_size, ' to be train'
    print 'sample ', test_size, ' to be test'

    ###pseudo random
    random.seed(1)
    sam = sorted(random.sample(range(0,len(vec)), train_size+test_size))
    random.seed(1)
    test_sam = sorted(random.sample(sam, test_size))

    train_vec = []
    train_lab = []
    tmp_vec = []
    tmp_lab = []
    test_vec = []
    test_lab = []

    random.seed(1)
    for i in xrange(len(vec)):
        v = vec[i]
        l = lab[i]
        r = random.randrange(1,90)
        if r == 1:
            tmp_vec.append(v)
            tmp_lab.append(l)
        else:
            train_vec.append(v)
            train_lab.append(l)
        if i % 100000 == 0:
            print "    at index ", i
    print "Done"
    print "train vec of size: ", len(train_vec)
    print "tmp vec of size: ", len(tmp_vec)

    print "sample from tmp vec of size ", len(tmp_vec)
    random.seed(1)
    sam = sorted(random.sample(range(0,len(tmp_vec)), test_size))
    print "sam of size ",len(sam)

    for i in xrange(len(tmp_vec)):
        if i % 10000 == 0:
            print "    at index ", i
        v = tmp_vec[i]
        l = tmp_lab[i]
        if len(test_vec) < test_size and i == sam[0]:
            test_vec.append(v)
            test_lab.append(l)
            sam = sam[1:]
        else:
            train_vec.append(v)
            train_lab.append(l)


    print "Done"
    print "train vec of size: ", len(train_vec)
    print "test vec of size: ", len(test_vec)

    print "Writing to files"
    train_vec_file = open("train_vectors", "w")
    for b in train_vec:
        train_vec_file.write(','.join(b) + '\n')
    print "writing train vector"
    train_lab_file = open("train_labels", "w")
    for b in train_lab:
        train_lab_file.write(b + '\n')
    print "writing train label"
    test_vec_file = open("test_vectors", "w")
    for b in test_vec:
        test_vec_file.write(','.join(b) + '\n')
    print "writing test vector"
    test_lab_file = open("test_labels", "w")
    for b in test_lab:
        test_lab_file.write(b + '\n')
    print "writing test label"

def filter_data_check():
    print 'Reading raw data file ...'
    data100 = pkl.load(open('simple_vec', "r"))

    unique_dict = defaultdict(int)

    index = 0
    count = 0
    for data in data100:
    	if all([i=='3' for i in data[:90]]):
    		print data[-20:]
        string_data = str(data)
        if not unique_dict[string_data]:
            unique_dict[string_data] = 1
            count += 1
        index += 1
        if index % 100000 == 0:
            print '    at index ', index
    print 'Done'
    print 'Down size from ', index, ' to ', count

if __name__ == "__main__":
	vec,lab = filter_data()
	split_train_test(vec, lab, 990000, 10000)
	#filter_data_check()