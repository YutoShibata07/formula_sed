#! /bin/bash

umask 007
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=0 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir0/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=1 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir1/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=2 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir2/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=3 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir3/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=4 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir4/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=5 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir5/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=6 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir6/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=7 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir7/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=8 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir8/ --n_iter=2500 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=9 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data2/1021_2100_per16ms_100k/dir9/ --n_iter=2500 &
wait