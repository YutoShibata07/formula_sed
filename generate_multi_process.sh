#! /bin/zsh
output_dir=./data/
# Please adjust the number of processes and MKL_NUM_THREADS so that their product matches the number of logical cores.
MKL_NUM_THREADS=5 python3 ./main.py --workid=0 --savedir=$output_dir --n_iter=2500 --seed=0&
MKL_NUM_THREADS=5 python3 ./main.py --workid=1 --savedir=$output_dir --n_iter=2500 --seed=1&
# MKL_NUM_THREADS=5 python3 ./generate_audio.py --workid=2 --savedir=$output_dir --n_iter=2500 --seed=2&
# MKL_NUM_THREADS=5 python3 ./generate_audio.py --workid=3 --savedir=$output_dir --n_iter=2500 --seed=3&
# MKL_NUM_THREADS=5 python3 ./generate_audio.py --workid=4 --savedir=$output_dir --n_iter=2500 --seed=4&
# MKL_NUM_THREADS=5 python3 ./generate_audio.py --workid=5 --savedir=$output_dir --n_iter=2500 --seed=5&
wait