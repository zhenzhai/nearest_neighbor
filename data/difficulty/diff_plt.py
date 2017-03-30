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

def plt_diff_value():
	font = {'size' : 20}
	plt.rc('font', **font)
	x = np.linspace(0,100)
	y = np.linspace(0,1)
	plt.ylim(0,0.6)
	X,Y = np.meshgrid(x,y)
	plt.title('Difficulty of Datasets', y=1.02)
	plt.xlabel('Percentile', labelpad = 10)
	plt.ylabel('Difficulty', labelpad = 10)

	filename = 'diff_percentiles.dat'
	with open(filename) as f:
		file_lines = f.readlines()
	y = [25, 50, 75]
	big5x = [f.split()[1] for f in file_lines[:3]]
	cifarx = [f.split()[1] for f in file_lines[3:6]]
	mnistx = [f.split()[1] for f in file_lines[6:9]]
	songsx = [f.split()[1] for f in file_lines[9:12]]
	w2vx = [f.split()[1] for f in file_lines[12:]]
	big5, = plt.plot(y, big5x, 'ro-', label='big5', lw=3, ms=8)
	cifar, = plt.plot(y, cifarx, 'co-', label='cifar', lw=3, ms=8)
	mnist, = plt.plot(y, mnistx, 'bo-', label='mnist', lw=3, ms=8)
	songs, = plt.plot(y, songsx, 'go-', label='songs', lw=3, ms=8)
	w2v, = plt.plot(y, w2vx, 'mo-', label='w2v', lw=3, ms=8)

	plt.legend(handles=[mnist, cifar, songs, big5, w2v],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("percentiles.png")


def plt_points():
	font = {'size' : 20}
	plt.rc('font', **font)
	x = np.linspace(0,1000)
	y = np.linspace(0,1)
	plt.ylim(0,1)
	X,Y = np.meshgrid(x,y)
	plt.title('Difficulty of Datasets', y=1.02)
	plt.xlabel('Index of Data Points', labelpad = 10)
	plt.ylabel('Difficulty', labelpad = 10)

	mnist = plt_diff('mnist', 'ro')
	cifar = plt_diff('cifar', 'co')
	songs = plt_diff('songs', 'bo')
	big5 = plt_diff('big5', 'go')
	w2v = plt_diff('w2v', 'mo')

	plt.legend(handles=[mnist, cifar, songs, big5, w2v],loc=4)
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

def plt_kd():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('K-D Tree True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)
	plt.axis([0,50000,0,0.8])

	mnist = kd('mnist', 'bo')
	cifar = kd('cifar', 'co')
	songs = kd('songs', 'go')
	big5 = kd('big5', 'ro')
	w2v = kd('w2v', 'mo')

	plt.legend(handles=[mnist, cifar, songs, big5, w2v],loc=1)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("all_kd.png")


#plt_diff_value()
plt_kd()