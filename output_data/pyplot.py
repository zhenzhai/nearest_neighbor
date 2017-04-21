import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def read_file(dataset, x_index, y_index, filename):
	with open('{0}/true_nn_accuracy/{1}'.format(dataset,filename)) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines]
		y = [float(i[y_index]) for i in file_lines]
	return x, y

def read_partial_file(dataset, x_index, y_index, start, end, skip, filename):
	with open('{0}/true_nn_accuracy/{1}'.format(dataset,filename)) as f:
		file_lines = [[i for i in line.split()] for line in f][1:]
		x = [float(i[x_index]) for i in file_lines[start:end:skip]]
		y = [float(i[y_index]) for i in file_lines[start:end:skip]]
	return x, y

def read_partial_file_w_label(dataset, x_index, y_index, label_index, start, end, skip, filename):
	with open('{0}/true_nn_accuracy/{1}'.format(dataset,filename)) as f:
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

def kd(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd.png".format(dataset))

def kd_pca(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(dataset,3,2,'pca_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line,pca_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_pca.png".format(dataset))

def kd_pca_rkd(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(dataset,3,2,'pca_tree.dat')
	rkd_x, rkd_y = read_file(dataset,3,2,'2rkd_tree.dat')
	rkd4_x, rkd4_y = read_file(dataset,3,2,'4rkd_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	rkd_line, = plt.plot(rkd_x, rkd_y, 'go-', label='2 R-K-D Trees', lw=3, ms=8)
	rkd4_line, = plt.plot(rkd4_x, rkd4_y, 'co-', label='4 R-K-D Trees', lw=3, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'mo-', label='8 R-K-D Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,pca_line, rkd_line,rkd4_line,rkd8_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_pca_rkd8.png".format(dataset))

def kd_rkd2(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	rkd_x, rkd_y = read_file(dataset,3,2,'2rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	rkd_line, = plt.plot(rkd_x, rkd_y, 'bo-', label='2 R-K-D Tree', lw=3, ms=8)
	plt.legend(handles=[kd_line,rkd_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_rkd2.png".format(dataset))

def kd_rkd8(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	rkd_x, rkd_y = read_file(dataset,3,2,'2rkd_tree.dat')
	rkd4_x, rkd4_y = read_file(dataset,3,2,'4rkd_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	rkd_line, = plt.plot(rkd_x, rkd_y, 'go-', label='2 R-K-D Trees', lw=3, ms=8)
	rkd4_line, = plt.plot(rkd4_x, rkd4_y, 'bo-', label='4 R-K-D Trees', lw=3, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'mo-', label='8 R-K-D Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,rkd_line,rkd4_line,rkd8_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_rkd8.png".format(dataset))

def kd_pca_rkd_rp(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(dataset,3,2,'pca_tree.dat')
	rp8_x, rp8_y = read_file(dataset,3,2,'8rp_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rkd_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'co-', label='8 RP Trees', lw=3, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'mo-', label='8 R-K-D Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line, pca_line, rp8_line, rkd8_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_pca_rkd8_rp8.png".format(dataset))

def kd_rp2(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	rp_x, rp_y = read_file(dataset,3,2,'2rp_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	rp_line, = plt.plot(rp_x, rp_y, 'bo-', label='2 RP Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,rp_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_rp2.png".format(dataset))

def kd_rp8(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	rp_x, rp_y = read_file(dataset,3,2,'2rp_tree.dat')
	rp4_x, rp4_y = read_file(dataset,3,2,'4rp_tree.dat')
	rp8_x, rp8_y = read_file(dataset,3,2,'8rp_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	rp_line, = plt.plot(rp_x, rp_y, 'go-', label='2 RP Trees', lw=3, ms=8)
	rp4_line, = plt.plot(rp4_x, rp4_y, 'bo-', label='4 RP Trees', lw=3, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'mo-', label='8 RP Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line,rp_line,rp4_line,rp8_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_rp8.png".format(dataset))

def kd_pca_rkd_rp_v2(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	pca_x, pca_y = read_file(dataset,3,2,'pca_tree.dat')
	rp8_x, rp8_y = read_file(dataset,3,2,'8rp_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rkd_tree.dat')
	v28_x, v28_y = read_file(dataset,3,2,'8v2_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	pca_line, = plt.plot(pca_x, pca_y, 'bo-', label='PCA Tree', lw=3, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'co-', label='8 RP Trees', lw=3, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'mo-', label='8 R-K-D Trees', lw=3, ms=8)
	v28_line, = plt.plot(v28_x, v28_y, 'go-', label='8 $V^2$ Trees', lw=3, ms=8)
	plt.legend(handles=[kd_line, pca_line, rp8_line, rkd8_line, v28_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_pca_rkd8_rp8_v28.png".format(dataset))

def kd_spill(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	kd_x, kd_y = read_file(dataset,3,2,'kd_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(dataset,4,3,5,0,25,2,'kd_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(dataset,4,3,5,1,25,2,'kd_spill_tree.dat')
	kd_line, = plt.plot(kd_x, kd_y, 'ro-', label='K-D Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='K-D Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='K-D Spill $\\alpha=0.1$', lw=3, ms=8)
	#label_points(spill05_x, spill05_y, -30, -60, spill05_label)
	#label_points(spill1_x, spill1_y, -50, 50, spill1_label)
	plt.legend(handles=[kd_line, spill05_line, spill1_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_kd_spill.png".format(dataset))

def pca_spill(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	pca_x, pca_y = read_file(dataset, 3,2,'pca_tree.dat')
	spill05_x, spill05_y, spill05_label = read_partial_file_w_label(dataset, 4,3,5,0,25,2,'pca_spill_tree.dat')
	spill1_x, spill1_y, spill1_label = read_partial_file_w_label(dataset, 4,3,5,1,25,2,'pca_spill_tree.dat')
	pca_line, = plt.plot(pca_x, pca_y, 'ro-', label='PCA Tree', lw=3, ms=8)
	spill05_line, = plt.plot(spill05_x, spill05_y, 'mo-', label='PCA Spill $\\alpha=0.05$', lw=3, ms=8)
	spill1_line, = plt.plot(spill1_x, spill1_y, 'go-', label='PCA Spill $\\alpha=0.1$', lw=3, ms=8)
	#label_points(spill05_x, spill05_y, 10, -60, spill05_label)
	#label_points(spill1_x, spill1_y, -30, 40, spill1_label)
	plt.legend(handles=[pca_line, spill05_line, spill1_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_pca_spill.png".format(dataset))

def spill_vspill(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	spill_x, spill_y, spill_label = read_partial_file_w_label(dataset,4,3,5,4,20,2,'kd_spill_tree.dat')
	vspill_x, vspill_y, vspill_label = read_partial_file_w_label(dataset,4,3,5,0,10,2,'kd_v_spill_tree.dat')
	spill_line, = plt.plot(spill_x, spill_y, 'ro-', label='K-D Spill $\\alpha=0.05$', lw=3, ms=8)
	vspill_line, = plt.plot(vspill_x, vspill_y, 'bo-', label='K-D Virtual Spill $\\alpha=0.05$', lw=3, ms=8)
	label_points(spill_x, spill_y, -10, 60, spill_label)
	label_points(vspill_x, vspill_y, 10, -40, vspill_label)
	plt.legend(handles=[spill_line,vspill_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_05spill_vspill.png".format(dataset))

def rkd_spill_vspill(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	spill_x, spill_y, spill_label = read_partial_file_w_label(dataset,4,3,5,1,18,2,'kd_spill_tree.dat')
	vspill_x, vspill_y, vspill_label = read_partial_file_w_label(dataset,4,3,5,1,10,2,'kd_v_spill_tree.dat')
	rkd_x, rkd_y = read_partial_file(dataset,3,2,0,5,1,'2rkd_tree.dat')
	rkd4_x, rkd4_y = read_partial_file(dataset,3,2,0,3,1,'4rkd_tree.dat')
	spill_line, = plt.plot(spill_x, spill_y, 'ro-', label='K-D Spill $\\alpha=0.1$', lw=3, ms=8)
	vspill_line, = plt.plot(vspill_x, vspill_y, 'bo-', label='K-D Virtual Spill $\\alpha=0.1$', lw=3, ms=8)
	rkd_line, = plt.plot(rkd_x, rkd_y, 'go-', label='2 r-k-d Trees', lw=3, ms=8)
	rkd4_line, = plt.plot(rkd4_x, rkd4_y, 'mo-', label='4 r-k-d Trees', lw=3, ms=8)
	label_points(spill_x, spill_y, -20, 70, spill_label)
	label_points(vspill_x, vspill_y, 10, -40, vspill_label)
	plt.legend(handles=[spill_line,vspill_line,rkd_line,rkd4_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_multi_spill_vspill.png".format(dataset))

def rp_v2(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	rkd_x, rkd_y = read_file(dataset,3,2,'2rp_tree.dat')
	rkd4_x, rkd4_y = read_file(dataset,3,2,'4rp_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rp_tree.dat')
	v2_x, v2_y = read_file(dataset,3,2,'2v2_tree.dat')
	v24_x, v24_y = read_file(dataset,3,2,'4v2_tree.dat')
	v28_x, v28_y = read_file(dataset,3,2,'8v2_tree.dat')
	rkd_line, = plt.plot(rkd_x, rkd_y, 'bo-', label='2 RP Trees', lw=3, ms=8)
	rkd4_line, = plt.plot(rkd4_x, rkd4_y, 'go-', label='4 RP Trees', lw=3, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'ro-', label='8 RP Trees', lw=3, ms=8)
	v2_line, = plt.plot(v2_x, v2_y, 'co-', label='2 $V^2$ Trees', lw=3, ms=8)
	v24_line, = plt.plot(v24_x, v24_y, 'mo-', label='4 $V^2$ Trees', lw=3, ms=8)
	v28_line, = plt.plot(v28_x, v28_y, 'ko-', label='8 $V^2$ Trees', lw=3, ms=8)
	plt.legend(handles=[rkd_line,rkd4_line,rkd8_line,v2_line,v24_line,v28_line],loc=legend_loc)
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_rp_v2.png".format(dataset))

def rkd_rp_v2(dataset, xran, yran, legend_loc):
	plt.axis([0,xran,0,yran])
	rp_x, rp_y = read_file(dataset,3,2,'2rp_tree.dat')
	rp4_x, rp4_y = read_file(dataset,3,2,'4rp_tree.dat')
	rp8_x, rp8_y = read_file(dataset,3,2,'8rp_tree.dat')
	v2_x, v2_y = read_file(dataset,3,2,'2v2_tree.dat')
	v24_x, v24_y = read_file(dataset,3,2,'4v2_tree.dat')
	v28_x, v28_y = read_file(dataset,3,2,'8v2_tree.dat')
	rkd_x, rkd_y = read_file(dataset,3,2,'2rkd_tree.dat')
	rkd4_x, rkd4_y = read_file(dataset,3,2,'4rkd_tree.dat')
	rkd8_x, rkd8_y = read_file(dataset,3,2,'8rkd_tree.dat')
	rp_line, = plt.plot(rp_x, rp_y, 'bo-', label='2 RP Trees', lw=2, ms=8)
	rp4_line, = plt.plot(rp4_x, rp4_y, 'bs-', label='4 RP Trees', lw=2, ms=8)
	rp8_line, = plt.plot(rp8_x, rp8_y, 'b^-', label='8 RP Trees', lw=2, ms=8)
	v2_line, = plt.plot(v2_x, v2_y, 'ro-', label='2 $V^2$ Trees', lw=2, ms=8)
	v24_line, = plt.plot(v24_x, v24_y, 'rs-', label='4 $V^2$ Trees', lw=2, ms=8)
	v28_line, = plt.plot(v28_x, v28_y, 'r^-', label='8 $V^2$ Trees', lw=2, ms=8)
	rkd_line, = plt.plot(rkd_x, rkd_y, 'go-', label='2 R-K-D Trees', lw=2, ms=8)
	rkd4_line, = plt.plot(rkd4_x, rkd4_y, 'gs-', label='4 R-K-D Trees', lw=2, ms=8)
	rkd8_line, = plt.plot(rkd8_x, rkd8_y, 'g^-', label='8 R-K-D Trees', lw=2, ms=8)
	plt.legend(handles=[rp_line,rp4_line,rp8_line,rkd_line,rkd4_line,rkd8_line,v2_line,v24_line,v28_line],loc=legend_loc, prop={'size':13})
	figure = plt.gcf()
	figure.set_size_inches(13, 10)
	plt.savefig("{0}/graphs/{0}_rkd_rp_v2.png".format(dataset))

def songs():
	font = {'size' : 18}
	plt.rc('font', **font)
	plt.title('Million Songs True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "songs"
	y_range = 35000
	#kd(data,y_range,1,2)
	#kd_pca(data,y_range,1,2)
	#kd_pca_rkd(data,y_range,1,2)
	#kd_rkd2(data,y_range,1,2)
	#kd_rkd8(data,y_range,1,2)
	#kd_pca_rkd_rp(data,y_range,1,2)
	#kd_rp2(data,y_range,1,2)
	#kd_rp8(data,y_range,1,2)
	#kd_pca_rkd_rp_v2(data,y_range,1,2)
	#rp_v2(data,y_range,1,2)
	#rkd_rp_v2(data,y_range,1,2)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

def sift():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('SIFT True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "sift"
	y_range = 35000
	#kd(data,y_range,1,4)
	#kd_pca(data,y_range,1,2)
	#kd_pca_rkd(data,y_range,1,4)
	#kd_rkd2(data,y_range,1,2)
	#kd_rkd8(data,y_range,1,4)
	#kd_pca_rkd_rp(data,y_range,1,4)
	#kd_rp2(data,y_range,1,2)
	#kd_rp8(data,y_range,1,4)
	#kd_pca_rkd_rp_v2(data,y_range,1,4)
	#rp_v2(data,y_range,1,4)
	#rkd_rp_v2(data,y_range,1,4)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

def cifar():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('CIFAR True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "cifar"
	y_range = 7000
	#kd(data,y_range,1,4)
	#kd_pca(data,y_range,1,4)
	#kd_pca_rkd(data,y_range,1,4)
	#kd_rkd2(data,y_range,1,4)
	#kd_rkd8(data,y_range,1,4)
	#kd_pca_rkd_rp(data,y_range,1,4)
	#kd_rp2(data,y_range,1,2)
	#kd_rp8(data,y_range,1,4)
	#kd_pca_rkd_rp_v2(data,y_range,1,4)
	#rp_v2(data,y_range,1,4)
	#rkd_rp_v2(data,y_range,1,4)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

def big5():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('BIG5 True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "big5"
	y_range = 35000
	#kd(data,y_range,1,4)
	#kd_pca(data,y_range,1,4)
	#kd_pca_rkd(data,y_range,1,2)
	#kd_rkd2(data,y_range,1,2)
	#kd_rkd8(data,y_range,1,2)
	#kd_pca_rkd_rp(data,y_range,1,2)
	#kd_rp2(data,y_range,1,2)
	#kd_rp8(data,y_range,1,2)
	#kd_pca_rkd_rp_v2(data,y_range,1,2)
	#rp_v2(data,y_range,1,2)
	#rkd_rp_v2(data,y_range,1,2)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

def w2v():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('Google News Word2vec True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "w2v"
	y_range = 100000
	#kd(data,y_range,1,4)
	#kd_pca(data,y_range,1,4)
	#kd_pca_rkd(data,y_range,1,2)
	#kd_rkd2(data,y_range,1,2)
	#kd_rkd8(data,y_range,1,2)
	#kd_pca_rkd_rp(data,y_range,1,2)
	#kd_rp2(data,y_range,1,2)
	#kd_rp8(data,y_range,1,2)
	#kd_pca_rkd_rp_v2(data,y_range,1,2)
	#rp_v2(data,y_range,1,2)
	#rkd_rp_v2(data,y_range,1,2)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

def mnist():
	font = {'size' : 20}
	plt.rc('font', **font)
	plt.title('MNIST True NN Percentage', y=1.02)
	plt.xlabel('Number of Distance Computations', labelpad = 10)
	plt.ylabel('Fraction Correct NN', labelpad = 10)

	data = "mnist"
	y_range = 8000
	#kd(data,y_range,1,4)
	#kd_pca(data,y_range,1,4)
	#kd_pca_rkd(data,y_range,1,4)
	#kd_rkd2(data,y_range,1,4)
	#kd_rkd8(data,y_range,1,4)
	#kd_pca_rkd_rp(data,y_range,1,4)
	#kd_rp2(data,y_range,1,4)
	#kd_rp8(data,y_range,1,4)
	#kd_pca_rkd_rp_v2(data,y_range,1,4)
	#rp_v2(data,y_range,1,4)
	#rkd_rp_v2(data,y_range,1,4)
	#kd_spill(data,y_range,1,4)
	#pca_spill(data,y_range,1,4)
	#spill_vspill(data,y_range,1,4)
	#rkd_spill_vspill(data,y_range,1,4)

songs()

