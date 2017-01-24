from collections import defaultdict
import random

def remove_duplicate(data100):
    print 'remove duplicate from data of size ', len(data100)
    big5_vec = []
    big5_lab = []
    unique_dict = defaultdict(int)

    index = 0
    count = 0
    for line in data100:
        data = line[:109]
        string_data = str(data)
        if not unique_dict[string_data]:
            unique_dict[string_data] = 1
            big5_vec.append(line[:100])
            big5_lab.append(''.join(line[104:109]))
            count += 1
        index += 1
        if index % 100000 == 0:
            print '    at index ', index
    print 'Done'
    print 'Down size from ', index, ' to ', count
    return big5_vec, big5_lab


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

    big5_train_vec = []
    big5_train_lab = []
    big5_test_vec = []
    big5_test_lab = []
    for i in xrange(len(vec)):
        if len(sam) == 0:
            break
        v = vec[i]
        l = lab[i]
        if i == sam[0]:
            if len(test_sam) == 0:
                big5_train_vec.append(v)
                big5_train_lab.append(l)
            elif i != test_sam[0]:
                big5_train_vec.append(v)
                big5_train_lab.append(l)
            else:
                big5_test_vec.append(v)
                big5_test_lab.append(l)
                test_sam = test_sam[1:]
            sam = sam[1:]
        if i % 100000 == 0:
            print "    at index", i
    print "Done"
    print "train vec of size: ", len(big5_train_vec)
    print "test vec of size: ", len(big5_test_vec)

    print "Writing to files"
    train_vec = open("convert_data/train_vectors", "w")
    for b in big5_train_vec:
        train_vec.write(','.join(b) + '\n')
    print "writing train vector"
    train_lab = open("convert_data/train_labels", "w")
    for b in big5_train_lab:
        train_lab.write(b + '\n')
    print "writing train label"
    test_vec = open("convert_data/test_vectors", "w")
    for b in big5_test_vec:
        test_vec.write(','.join(b) + '\n')
    print "writing test vector"
    test_lab = open("convert_data/test_labels", "w")
    for b in big5_test_lab:
        test_lab.write(b + '\n')
    print "writing test label"

if __name__ == '__main__':
    print 'Reading raw data file ...'
    big5_file = open('big5.csv', 'r')
    big5 = [b.split(',')[1:] for b in big5_file.readlines()]

    ### Only capture data with 100 questions
    print 'Filtering raw data ...'
    big5_100data = []
    for b in big5[1:]:
        if b[101] == "100":
            big5_100data.append(b)

    vec, lab = remove_duplicate(big5_100data)
    split_train_test(vec, lab, 10000, 1000)




