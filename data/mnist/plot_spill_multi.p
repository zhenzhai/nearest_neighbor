#plot multi_tree and spill_tree data
set term aqua font "Helvetica,25"
set title "MNIST True NN Percentage" offset 0, -1
set xlabel "# of distance computations" offset 0, 1.3
set ylabel "Fraction Correct NN" offset 2, 0
set xtics offset 0, 0.5
set xrange[0:15000]
plot "kd_spill_tree.dat" every 2::2::20 u 5:4 title 'spill alpha 0.1' w linespoints lw 3,\
	 "kd_v_spill_tree.dat" every 2::2::20 u 5:4 title 'virtual spill alpha 0.1' w linespoints lw 3,\
     "multi_kd_tree.dat" every ::1::8 u 4:3 title 'Multiple Trees' w linespoints lw 3
