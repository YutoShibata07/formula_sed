# Formula-Supervised Sound Event Detection: Pre-Training Without Real Data (ICASSP2025)
![Overview](images/data_comparison_0331.png)
This repo contains the official implementation of [Y.Shibata et al. "Formula-Supervised Sound Event Detection: Pre-Training Without Real Data" in ICASSP2025](https://yutoshibata07.github.io/Formula-SED/).

The aim is to facilitate research and development by offering synthetic data that can be used in various audio-related machine learning workflows.

## Current Status

- [x] **Artificial Acoustic Data Generation**: The code for generating the proposed artificial acoustic data for pre-training is publicly available.
- [ ] **Pre-training Code**: The code for pre-training will be released in the future.

## Requirements

* Python >= 3.9
* GPy
* Librosa

## Directory Structure
```directory structure
root
├── constants.py
├── environment.yml
├── generate_audio.py
├── generate_multi_process.sh
├── helper.py
├── images
├── README.md
├── synth_params.py
├── utils.py
└── data
    ├── label
    ├── targets
    ├── volumes
    └── wav
```
* `constants.py` is for storing common constants, kernel definitions, and boundary settings used by the modeling process.
* `generate_audio.py` is a script for audio chunk generation.
* `generate_multi_process.sh` is for accelerated data generation through parallel processing.
* `helper.py` contains some functions about sampling, labeling, and audio synthesis.
* `synth_params.py` is for synthesis hyperparameters such as sample rate.
* `data/label` contains generated multi-hot labels for each audio file.
* `data/targets` contains generated continuous labels (without discretization) for each audio file.
* `data/volumes` contains volumens of generated audio which are used for post-processing.
* `data/wav` contains generated audio files.


## Installation
Our data generation process is highly CPU-intensive, so we use `Intel MKL` to accelerate it. To ensure smooth integration of Intel MKL, we use conda as our virtual environment manager.
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/YutoShibata07/formula_sed.git
cd formula_sed
mamba env create -f environment.yml
mamba activate formula-sed
```
Please check whether NumPy is using `Intel MKL` as its BLAS/LAPACK backend.
```bash
python -c "import numpy as np; np.show_config()"
```
You will see "mkl-sdl" as the output.
## Data Generation (single Process)
```bash
python3 ./generate_audio.py --workid=0 --savedir=data --n_iter=2 --seed=0 
```
For reproducibility, please verify that the two audio files in the `sample_audios/` directory have been successfully generated in the `data/` directory. 
## Data Generation (Multi Process)
By using `Intel MKL`, you can limit the number of threads used by each process to prevent conflicts between processes. Please adjust the number of parallel processes according to the number of available CPU cores in your environment.
```bash
sh generate_multi_process.sh
```
Stay tuned for updates regarding the release of the pre-training code. We appreciate your interest in this project!

## License

This repository is released under the MIT License.

## Citation
If you use the results of this project, please cite the following reference:
```bibtex
@INPROCEEDINGS{10888414,
  author={Shibata, Yuto and Tanaka, Keitaro and Bando, Yoshiaki and Imoto, Keisuke and Kalaoka, Hirokatsu and Aoki, Yoshimitsu},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Formula-Supervised Sound Event Detection: Pre-Training Without Real Data}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Accuracy;Event detection;Noise;Supervised learning;Training data;Acoustics;Mathematical models;Timing;Synthetic data;sound event detection;pre-training without real data;environmental sound synthesis},
  doi={10.1109/ICASSP49660.2025.10888414}}

