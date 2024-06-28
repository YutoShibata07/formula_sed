#!/bin/sh
#$ -l rt_F=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -o /groups/gac50523/user/shibata/fdsl-sed
source /etc/profile.d/modules.sh
module load python/3.6/3.6.12
module load cuda/10.2/10.2.89
module load cudnn/7.6/7.6.5
module load nccl/2.7/2.7.8-1
module load gcc/7.4.0
export PATH="/home/acf15690kd/anaconda3/bin:${PATH}"

source activate sed-fdsl
echo 1
cd /groups/gac50523/user/shibata/fdsl-sed
umask 007
echo 2
python check_mkl.py
export MKL_NUM_THREADS=1
echo $MKL_NUM_THREADS
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=0 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir10/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=1 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir11/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=2 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir12/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=3 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir13/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=4 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir14/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=5 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir15/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=6 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir16/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=7 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir17/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=8 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir18/ --n_iter=10000 &
MKL_NUM_THREADS=5 python3 ./GPyDDSP_1021_2100.py --workid=9 --savedir=/groups/gac50523/user/shibata/fdsl-sed/data4/1021_2100_per16ms_1m/dir19/ --n_iter=10000 &
wait
