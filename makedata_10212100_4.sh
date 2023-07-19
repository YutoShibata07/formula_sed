#! /bin/bash

umask 007
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=0 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir0_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=1 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir1_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=2 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir2_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=3 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir3_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=4 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir4_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=5 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir5_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=6 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir6_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=7 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir7_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=8 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir8_c/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=9 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_100k/dir9_c/ --n_iter=2500 &
wait