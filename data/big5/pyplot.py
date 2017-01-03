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
	plt.axis([0,15000,0,0.45])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd.png")


def kd_pca():
	plt.axis([0,15000,0,0.7])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line,pca_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd_pca.png")

def kd_spill():
	plt.axis([0,14000,0,0.5])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(4,3,5,0,25,2,'kd_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(4,3,5,1,25,2,'kd_spill_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='KD Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='KD Spill $\\alpha=0.1$', lw=3, ms=8)
	#label_points(spill05_x, spill05_y, -30, -60, spill05_label)
	#label_points(spill1_x, spill1_y, -30, 40, spill1_label)
	plt.legend(handles=[kd_line, spill05_line, spill1_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd_spill.png")

def pca_spill():
	plt.axis([0,14000,0.2,0.8])
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(4,3,5,0,21,2,'pca_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(4,3,5,1,21,2,'pca_spill_tree.dat')
	pca_line, = plt.plot(pca_x, pca_y, 'ro-', label='PCA Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='PCA Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='PCA Spill $\\alpha=0.1$', lw=3, ms=8)
	#label_points(spill05_x, spill05_y, -30, -60, spill05_label)
	#label_points(spill1_x, spill1_y, -30, 40, spill1_label)
	plt.legend(handles=[pca_line, spill05_line, spill1_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_pca_spill.png")

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
	plt.savefig(data_set + "_1spill_vspill.png")

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
	plt.savefig(data_set + "_multi_spill_vspill.png")

def ran2_kd():
	plt.axis([0,15000,0,0.6])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	ran_x, ran_y = read_file(3,2,'2rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'bo-', label='2 RKD Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_ran2_kd.png")

def ran8_kd():
	plt.axis([0,15000,0,0.7])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	ran_x, ran_y = read_file(3,2,'2rkd_tree.dat')
	ran4_x, ran4_y = read_file(3,2,'4rkd_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'go-', label='2 RKD Trees', lw=3, ms=8)
	ran4_line, = plt.plot(ran4_x, ran4_y, 'bo-', label='4 RKD Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RKD Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line,ran4_line,ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_ran8_kd.png")

def kd_pca_ran():
	plt.axis([0,15000,0,0.6])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	ran_x, ran_y = read_file(3,2,'2rkd_tree.dat')
	ran4_x, ran4_y = read_file(3,2,'4rkd_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'go-', label='2 RKD Trees', lw=3, ms=8)
	ran4_line, = plt.plot(ran4_x, ran4_y, 'co-', label='4 RKD Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RKD Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line, pca_line, ran_line, ran4_line, ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd_pca_ran8.png")

def rp2_kd():
	plt.axis([0,15000,0,0.5])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	ran_x, ran_y = read_file(3,2,'2rp_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'bo-', label='2 RP Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_rp2_kd.png")

def rp8_kd():
	plt.axis([0,15000,0,0.6])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	ran_x, ran_y = read_file(3,2,'2rp_tree.dat')
	ran4_x, ran4_y = read_file(3,2,'4rp_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rp_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	ran_line, = plt.plot(ran_x, ran_y, 'go-', label='2 RP Trees', lw=3, ms=8)
	ran4_line, = plt.plot(ran4_x, ran4_y, 'bo-', label='4 RP Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RP Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,ran_line,ran4_line,ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_rp8_kd.png")

def rp_rps():
	plt.axis([0,17000,0,0.8])
	ran_x, ran_y = read_file(3,2,'2rp_tree.dat')
	ran4_x, ran4_y = read_file(3,2,'4rp_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rp_tree.dat')
	rps_x, rps_y = read_file(3,2,'2rp_select_tree.dat')
	rps4_x, rps4_y = read_file(3,2,'4rp_select_tree.dat')
	rps8_x, rps8_y = read_file(3,2,'8rp_select_tree.dat')
	ran_line, = plt.plot(ran_x, ran_y, 'bo-', label='2 RP Trees', lw=3, ms=8)
	ran4_line, = plt.plot(ran4_x, ran4_y, 'go-', label='4 RP Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'ro-', label='8 RP Trees', lw=3, ms=8)
	rps_line, = plt.plot(rps_x, rps_y, 'co-', label='2 RP Select Trees', lw=3, ms=8)
	rps4_line, = plt.plot(rps4_x, rps4_y, 'mo-', label='4 RP Select Trees', lw=3, ms=8)
	rps8_line, = plt.plot(rps8_x, rps8_y, 'ko-', label='8 RP Select Trees', lw=3, ms=8)
	plt.legend(handles=[ran_line,ran4_line,ran8_line,rps_line,rps4_line,rps8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_rp_rps.png")

def kd_pca_ran_rp():
	plt.axis([0,14000,0,0.7])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	rp8_x, rp8_y = read_file(3,2,'8rp_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'co-', label='8 RP Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RKD Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line, pca_line, rp8_line, ran8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd_pca_ran8_rp8.png")

def kd_pca_ran_rp_rs():
	plt.axis([0,14000,0,0.8])
	kd_x, kd_y = read_file(3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(3,2,'pca_tree.dat')
	rp8_x, rp8_y = read_file(3,2,'8rp_tree.dat')
	ran8_x, ran8_y = read_file(3,2,'8rkd_tree.dat')
	rs8_x, rs8_y = read_file(3,2,'8rp_select_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='KD Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'co-', label='8 RP Trees', lw=3, ms=8)
	ran8_line, = plt.plot(ran8_x, ran8_y, 'mo-', label='8 RKD Trees', lw=3, ms=8)
	rs8_line, = plt.plot(rs8_x, rs8_y, 'go-', label='8 RP Select Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line, pca_line, rp8_line, ran8_line, rs8_line],loc=4)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig(data_set + "_kd_pca_ran8_rp8_rs8.png")


def main():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('BIG5 True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	global data_set
	data_set = 'big5'
	rp_rps()

main()

