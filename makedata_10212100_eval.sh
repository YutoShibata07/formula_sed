#! /bin/bash

umask 007
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=0 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir0/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=1 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir1/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=2 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir2/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=3 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir3/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=4 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir4/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=5 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir5/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=6 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir6/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=7 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir7/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=8 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir8/ --n_iter=2500 &
MKL_NUM_THREADS=5 python ./GPyDDSP_1021_2100.py --workid=9 --savedir=/groups/gaa50073/tanaka-keitaro/fdsl-sed/data/1021_2100_per16ms_25k/dir9/ --n_iter=2500 &
wait