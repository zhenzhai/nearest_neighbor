import random
from collections import defaultdict
import pickle as pkl

def sampling_data(train_size, test_size):
    print 'Start sampling ...'
    print 'Reading data and label'
    f = open('GoogleNewsData', 'r')
    lines_v = f.readlines()
    f.close()
    fl = open('GoogleNewsLabels', 'r')
    lines_l= fl.readlines()
    fl.close()


    print 'Create int labels'
    lab_dic = defaultdict(str)
    for l in lines_l:
    	if not lab_dic[l]:
    		lab_dic[l] = str(len(lab_dic))
    pkl.dump(lab_dic, open('label_mapping.pkl','w'))

    print 'sample from dataset of size ', len(lines_v)
    print 'sample ', train_size, ' to be train'
    print 'sample ', test_size, ' to be test'

    train_vec = []
    train_lab = []
    tmp_vec = []
    tmp_lab = []
    test_vec = []
    test_lab = []

    random.seed(1)
    for i in xrange(len(lines_v)):
        v = lines_v[i]
        l = lab_dic[lines_l[i]]
        r = random.randrange(1,100)
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
        train_vec_file.write(b)
    print "writing train vector"
    train_lab_file = open("train_labels", "w")
    for b in train_lab:
        train_lab_file.write(b + '\n')
    print "writing train label"
    test_vec_file = open("test_vectors", "w")
    for b in test_vec:
        test_vec_file.write(b)
    print "writing test vector"
    test_lab_file = open("test_labels", "w")
    for b in test_lab:
        test_lab_file.write(b + '\n')
    print "writing test label"


if __name__ == "__main__":
    sampling_data(2990000, 10000)
