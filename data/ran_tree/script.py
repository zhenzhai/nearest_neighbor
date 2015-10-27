import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def read_file(multi_file, kd_file):
	with open(multi_file) as f:
		file_lines = [[i for i in l.split()] for l in f][1:]
		multi_num = [float(i[2]) for i in file_lines]
		dist = [float(i[3]) for i in file_lines]
	with open(kd_file) as f:
		file_lines = [[i for i in l.split()] for l in f][1:]
		kd_num = [float(i[2]) for i in file_lines]
	diff = np.subtract(kd_num, multi_num)
	return dist, diff

def main():
	cifar_dist, cifar_diff = read_file('cifar_1multi_kd_tree.dat', 'cifar_kd_tree.dat')
	mnist_dist, mnist_diff = read_file('mnist_1multi_kd_tree.dat', 'mnist_kd_tree.dat')
	font = {'size' : 25}
	plt.rc('font', **font)
	plt.title('Difference Between KD Tree and Randomized KD Tree', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN Difference', labelpad = 10)
	plt.axis([0,8000,0.03,0.08])
	kd_line, = plt.plot(cifar_dist, cifar_diff, 'ro-', label='CIFAR', lw=3, ms=8)
	ran_line, = plt.plot(mnist_dist, mnist_diff, 'bo-', label='MNIST', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("diff.png")

main()