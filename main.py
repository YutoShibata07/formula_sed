#! /usr/bin/env python3

import os
import argparse
import numpy as np
import GPy
import soundfile as sf
import glob
import json

from libs.audio_generation import generate_one_sample
from libs.synth_params import SynthesisParams
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workid", default=0, type=int)
    parser.add_argument("--n_iter", default=0, type=int)
    parser.add_argument("--savedir", default=None, type=str)
    parser.add_argument("--seed", default=44, type=int)
    args = parser.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    savedir = args.savedir
    if os.path.isdir(savedir):
        print(f"Directory '{savedir}' already exists!")
        gen_counter = len(glob.glob(os.path.join(savedir, "wav/*.wav")))
        print(f"{args.workid} created {gen_counter} files already!")
    else:
        gen_counter = 0
    os.makedirs(f"{savedir}/" + "wav/", exist_ok=True)
    os.makedirs(f"{savedir}/" + "label/", exist_ok=True)
    os.makedirs(f"{savedir}/" + "targets/", exist_ok=True)
    os.makedirs(f"{savedir}/" + "volumes/", exist_ok=True)

    """(0) Fixed paramater settings
    """

    ### Default kernel
    default_kernel = GPy.kern.RBF

    syn_params = SynthesisParams()

    params_dict = syn_params.to_dict()
    with open(os.path.join(savedir, "synthesis_params.json"), "w") as f:
        json.dump(params_dict, f, indent=2)

    n_iter = args.n_iter

    while gen_counter < n_iter:
        n_to_mix = rng.integers(1, syn_params.max_n_to_mix + 1)
        summed_wav = np.zeros(syn_params.n_samples)
        raw_target_list = []
        volume_list = []
        summed_wav, summed_label, raw_target_list, volume_list = generate_one_sample(
            rng=rng,
            max_n_to_mix=syn_params.max_n_to_mix,
            n_samples=syn_params.n_samples,
            small_variance=syn_params.small_variance,
            duration=syn_params.duration,
            frames_per_sec=syn_params.frames_per_sec,
            n_frames=syn_params.n_frames,
            default_kernel=default_kernel,
            n_frequencies=syn_params.n_frequencies,
            noise_anchors=syn_params.noise_anchors,
            noise_vagueness=syn_params.noise_vagueness,
            sample_rate=syn_params.sample_rate,
            workid=args.workid,
        )

        # summed_wav /= n_to_mix
        try:
            assert summed_wav.max() < 1
        except Exception:
            print(f"Work id {args.workid}: Skipped (summed_wav.max() >= 1).")
            continue

        sf.write(
            f"{savedir}/" + f"wav/{gen_counter}_{n_to_mix}mix.wav",
            summed_wav,
            syn_params.sample_rate,
            "PCM_24",
        )
        np.save(f"{savedir}/" + f"label/{gen_counter}_{n_to_mix}mix.npy", summed_label)
        np.save(
            f"{savedir}/" + f"targets/{gen_counter}_{n_to_mix}mix.npy",
            np.array(raw_target_list),
        )
        np.save(
            f"{savedir}/" + f"volumes/{gen_counter}_{n_to_mix}mix.npy",
            np.array(volume_list),
        )

        gen_counter += 1
        if (100 * gen_counter) % n_iter == 0:
            print(
                f"Work id {args.workid}: {gen_counter}/{n_iter} ({(100 * gen_counter) // n_iter}%) done."
            )

    print(f"Work id {args.workid}: Finished. Genarated {gen_counter} samples in total.")


if __name__ == "__main__":
    main()
