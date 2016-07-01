#plot pca_tree and kd_tree 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def read_file(x_index, y_index, file):
	with open(file) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines]
		y = [float(i[y_index]) for i in file_lines]
	return x, y

def read_partial_file(x_index, y_index, start, end, skip, file):
	with open(file) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines[start:end:skip]]
		y = [float(i[y_index]) for i in file_lines[start:end:skip]]
	return x, y

def read_partial_file_w_label(x_index, y_index, label_index, start, end, skip, file):
	with open(file) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines[start:end:skip]]
		y = [float(i[y_index]) for i in file_lines[start:end:skip]]
		labels = [float(i[label_index]) for i in file_lines[start:end:skip]]
	return x, y, labels

def label_points(x_index, y_index, x_off, y_off, labels):
	for i in xrange(len(x_index)):
		plt.annotate(labels[i],
					 xy=(x_index[i],y_index[i]), 
					 xytext=(x_off, y_off), 
					 textcoords='offset points', 
					 arrowprops=dict(arrowstyle="->",
					 connectionstyle="arc,rad=10"))


def kd():
	plt.axis([0,7000,0.3,0.75])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_kd.png")


def kd_pca():
	plt.axis([0,7000,0.3,0.75])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line,pca_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_kd_pca.png")

def kd_spill():
	plt.axis([0,9000,0.3,0.9])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(4,3,5,0,19,2,'kd_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(4,3,5,1,19,2,'kd_spill_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='KD Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='KD Spill $\\alpha=0.1$', lw=3, ms=8)
	label_points(spill05_x, spill05_y, -30, -60, spill05_label)
	label_points(spill1_x, spill1_y, -30, 40, spill1_label)
	plt.legend(handles=[kd_line, spill05_line, spill1_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_kd_spill.png")

def pca_spill():
	plt.axis([0,9000,0.3,0.9])
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(4,3,5,0,21,2,'pca_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(4,3,5,1,21,2,'pca_spill_tree.dat')
	pca_line, = plt.plot(pca_x, pca_y, 'ro-', label='PCA Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='PCA Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='PCA Spill $\\alpha=0.1$', lw=3, ms=8)
	label_points(spill05_x, spill05_y, -30, -60, spill05_label)
	label_points(spill1_x, spill1_y, -30, 40, spill1_label)
	plt.legend(handles=[pca_line, spill05_line, spill1_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_pca_spill.png")

def spill_vspill():
	plt.axis([0,8000,0.25,0.85])
	spill_x, spill_y, spill_label = read_partial_file_w_label(4,3,5,1,13,2,'kd_spill_tree.dat')
	vspill_x, vspill_y, vspill_label = read_partial_file_w_label(4,3,5,1,20,2,'kd_v_spill_tree.dat')
	spill_line, = plt.plot(spill_x, spill_y, 'ro-', label='KD Spill $\\alpha=0.1$', lw=3, ms=8)
	vspill_line, = plt.plot(vspill_x, vspill_y, 'bo-', label='KD Virtual Spill $\\alpha=0.1$', lw=3, ms=8)
	label_points(spill_x, spill_y, 10, -60, spill_label)
	label_points(vspill_x, vspill_y, -40, 20, vspill_label)
	plt.legend(handles=[spill_line,vspill_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_1spill_vspill.png")

def ran_spill_vspill():
	plt.axis([0,10000,0.35,0.85])
	spill_x, spill_y, spill_label = read_partial_file_w_label(4,3,5,1,18,2,'kd_spill_tree.dat')
	vspill_x, vspill_y, vspill_label = read_partial_file_w_label(4,3,5,1,20,2,'kd_v_spill_tree.dat')
	ran_x, ran_y = read_partial_file(3,2,0,5,1,'multi_kd_tree.dat')
	ran4_x, ran4_y = read_partial_file(3,2,0,3,1,'4multi_kd_tree.dat')
	ran8_x, ran8_y = read_partial_file(3,2,0,5,1,'8multi_kd_tree.dat')	
	spill_line, = plt.plot(spill_x, spill_y, 'ro-', label='KD Spill $\\alpha=0.1$', lw=3, ms=8)
	vspill_line, = plt.plot(vspill_x, vspill_y, 'bo-', label='KD Virtual Spill $\\alpha=0.1$', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'go-', label='2 Multiple Trees', lw=3, ms=8)
	ran4_line, = plt.plot(ran4_x, ran4_y, 'mo-', label='4 Multiple Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'co-', label='8 Multiple Trees', lw=3, ms=8)
	label_points(spill_x, spill_y, 10, -60, spill_label)
	label_points(vspill_x, vspill_y, -60, 20, vspill_label)
	plt.legend(handles=[spill_line,vspill_line,ran_line, ran4_line, ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_multi_spill_vspill.png")

def ran_kd():
	plt.axis([0,7000,0.2,0.9])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	ran_x, ran_y = read_partial_file(3,2,0,3,1,'4multi_kd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'bo-', label='4 RP Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_ran4_kd.png")

def ran8_kd():
	plt.axis([0,7000,0.2, 0.9])
	kd_x, kd_y = read_partial_file(3,2,1,10,1,'kd_tree.dat')
	ran_x, ran_y = read_partial_file(3,2,0,3,1,'4multi_kd_tree.dat')
	ran8_x, ran8_y = read_partial_file(3,2,0,5,1,'8multi_kd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'bo-', label='4 RP Tree', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RP Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line, ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("cifar_ran8_kd.png")

def main():
	font = {'size' : 25}
	plt.rc('font', **font)
	plt.title('CIFAR True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	ran8_kd()

main()

