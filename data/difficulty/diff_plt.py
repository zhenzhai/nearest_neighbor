import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plt_diff(file_name, color):
	filename = file_name + '_difficulty.dat'
	with open(filename) as f:
		file_lines = f.readlines()[1:]
	x = [float(i) for i in file_lines]
	x = np.random.choice(x, 1000)
	dat, = plt.plot(x, color, label=file_name)
	return dat

def get_percentile(file_name, perc):
	print "getting percentile for ", file_name
	with open(file_name + '_difficulty.dat') as f:
		lines = np.array([d.strip() for d in f.readlines()[1:]])
	all_diff = lines.astype(np.float)
	perc_value = []
	for p in perc:
		print '		', p
		perc_value.append(np.percentile(all_diff, p))
	'''with open('diff_percentile.dat', 'a') as wf:
		wf.write(file_name)
		wf.write('\t')
		wf.write(str(perc_value))
		wf.write('\n')'''
	return perc_value

def plt_diff_value():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('Difficulty of Datasets', y=1.02)
	plt.xlabel('Percentile', labelpad = 10)
	plt.ylabel('Difficulty', labelpad = 10)
	plt.axis([0,100,0,0.6])

	y = [25, 50, 75]
	big5x = get_percentile('big5', y)
	cifarx = get_percentile('cifar', y)
	mnistx = get_percentile('mnist', y)
	siftx = get_percentile('sift', y)
	w2vx = get_percentile('w2v', y)
	songsx = get_percentile('songs', y)

	big5, = plt.plot(y, big5x, 'ro-', label='big5', lw=3, ms=8)
	cifar, = plt.plot(y, cifarx, 'co-', label='cifar', lw=3, ms=8)
	mnist, = plt.plot(y, mnistx, 'bo-', label='mnist', lw=3, ms=8)
	songs, = plt.plot(y, songsx, 'go-', label='songs', lw=3, ms=8)
	sift, = plt.plot(y, siftx, 'ko-', label='sift', lw=3, ms=8)
	w2v, = plt.plot(y, w2vx, 'mo-', label='w2v', lw=3, ms=8)

	plt.legend(handles=[mnist, cifar, songs, big5, w2v, sift],loc=2)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("percentiles.png")


def plt_points():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.ylim(0,1)
	plt.title('Difficulty of Datasets', y=1.02)
	plt.xlabel('Index of Data Points', labelpad = 10)
	plt.ylabel('Difficulty', labelpad = 10)

	mnist = plt_diff('mnist', 'ro')
	cifar = plt_diff('cifar', 'co')
	songs = plt_diff('songs', 'bo')
	big5 = plt_diff('big5', 'go')
	w2v = plt_diff('w2v', 'mo')
	sift = plt_diff('sift', 'ko')

	plt.legend(handles=[mnist, cifar, songs, big5, w2v, sift],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("diff.png")

def read_file(x_index, y_index, file):
	with open(file) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines]
		y = [float(i[y_index]) for i in file_lines]
	return x, y

def kd(data_set, color):
	kd_x, kd_y = read_file(3,2,'../{0}/true_nn_accuracy/kd_tree.dat'.format(data_set))
	kd_line, = plt.plot(kd_x, kd_y, color+'-', label=data_set, lw=3, ms=8)
	return kd_line

def v2(data_set, color):
	v2_x, v2_y = read_file(3,2,'../{0}/true_nn_accuracy/8v2_tree.dat'.format(data_set))
	v2_line, = plt.plot(v2_x, v2_y, color+'-', label=data_set, lw=3, ms=8)
	return v2_line

def plt_all(tree, yran):
	font = {'size' : 20}
	plt.rc('font', **font)
	if tree == "kd":
		tree_name = "K-D"
	elif tree == "v2":
		tree_name = "Two Vantage Point"
	plt.title(tree_name+' Tree True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)
	plt.axis([0,50000,0,yran])

	if tree == 'kd':
		mnist = kd('mnist', 'bo')
		cifar = kd('cifar', 'co')
		songs = kd('songs', 'go')
		big5 = kd('big5', 'ro')
		w2v = kd('w2v', 'mo')
		sift = kd('sift', 'ko')
	elif tree == 'v2':
		mnist = v2('mnist', 'bo')
		cifar = v2('cifar', 'co')
		songs = v2('songs', 'go')
		big5 = v2('big5', 'ro')
		w2v = v2('w2v', 'mo')
		sift = v2('sift', 'ko')

	plt.legend(handles=[mnist, cifar, songs, big5, w2v, sift],loc=1)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("all_{0}.png".format(tree))




plt_diff_value()
#plt_all("kd", 1)
#plt_all("v2", 1)