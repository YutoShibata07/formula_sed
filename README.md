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
├── data
│   ├── label
│   ├── synthesis_params.json
│   ├── targets
│   ├── volumes
│   └── wav
├── ddsp
├── environment.yml
├── generate_multi_process.sh
├── images
├── libs
│   ├── audio_generation.py
│   ├── constants.py
│   ├── duration.py
│   ├── kernels.py
│   ├── labeling.py
│   ├── synth_params
│   └── utils.py
├── main.py
├── README.md
└── sample_audios
```
* `data` is for saving the generated audio file, labels (discrete/continuous).
* `generate_multi_process.sh` is for accelerated data generation through parallel processing.
* `lib/audio_generation.py` is for single audio file generation.
* `libs/constants.py` is for storing common constants (boundary settings used by the modeling process).
* `lib/duration.py` samples random duration.
* `lib/kernels.py` samples kernel types used for global/local F0/volume variation.
* `lib/labeling.py` is for discrete label generation.
* `synth_params` is for dataclass of global/local synthesis parameters.
* `sample_audios` are used for test.


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
You will see `libraries = ['mkl_rt', 'pthread']` as the output.
## Data Generation (single Process)
```bash
python3 ./main.py --workid=0 --savedir=data --n_iter=2 --seed=0 
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
```
## Reference

* Jesse Engel et al., "DDSP: Differentiable Digital Signal Processing", in ICLR2020 ([paper](https://arxiv.org/abs/2001.04643))