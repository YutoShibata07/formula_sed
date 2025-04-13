import numpy as np
from typing import List, Tuple


def sample_voiced_silence_durations(
    rng: np.random.Generator,
    repeat_times: int,
    scale_exp_silent: float,
    scale_exp_voiced: float,
    scale_normal_silent: float,
    scale_normal_voiced: float,
    duration: float,
    frames_per_sec: int,
    n_frames: int,
) -> Tuple[List[int], List[int]]:
    """
    Sample the lengths of voiced and silence segments, measured in frames,
    and return both as a tuple.

    This function uses exponential distributions for the initial silent and
    voiced segments, and then uses normal distributions for subsequent segments
    while ensuring that durations remain positive. Finally, it appends an
    additional silent segment specified by `duration`.

    Args:
        rng: A random number generator (NumPy's >=1.17 recommended interface).
        repeat_times: The number of times to alternate between a silent
            segment and a voiced segment.
        scale_exp_silent: The scale parameter for the exponential distribution
            used to sample the first silent segment (in seconds).
        scale_exp_voiced: The scale parameter for the exponential distribution
            used to sample the first voiced segment (in seconds).
        scale_normal_silent: The standard deviation for the normal distribution
            used to sample subsequent silent segments (in seconds).
        scale_normal_voiced: The standard deviation for the normal distribution
            used to sample subsequent voiced segments (in seconds).
        duration: An additional duration in seconds to be appended as a final
            silent segment after all repeats.
        frames_per_sec: The number of frames per second (typically sample_rate // hop_size).
        n_frames: The maximum total number of frames allowed.

    Returns:
        A tuple (voiced_durations, silent_durations), where each is a list of
        segment lengths in frames. The final total of frames is capped at `n_frames`.
    """
    # verbose arrays
    _silent_durations = []
    _voiced_durations = []

    # set by sec
    for i in range(repeat_times):
        if i == 0:
            _silent_durations.append(rng.exponential(scale_exp_silent))
            _voiced_durations.append(rng.exponential(scale_exp_voiced))

        else:
            while True:
                candidate_silent_duration = rng.normal(
                    _silent_durations[-1], scale_normal_silent
                )
                if candidate_silent_duration > 0:
                    break
            _silent_durations.append(candidate_silent_duration)

            while True:
                candidate_voiced_duration = rng.normal(
                    _voiced_durations[-1], scale_normal_voiced
                )
                if candidate_voiced_duration > 0:
                    break
            _voiced_durations.append(candidate_voiced_duration)

    # padding
    _silent_durations.append(duration)

    # change to frames
    _silent_durations = [int(elm * frames_per_sec) for elm in _silent_durations]
    _voiced_durations = [int(elm * frames_per_sec) for elm in _voiced_durations]

    # arrays extracted from verbose arrays
    silent_durations = []
    voiced_durations = []
    total_frames = 0

    for i in range(repeat_times + 1):
        silent_durations.append(_silent_durations[i])
        total_frames += _silent_durations[i]
        if total_frames >= n_frames:
            silent_durations[-1] -= total_frames - n_frames
            break

        voiced_durations.append(_voiced_durations[i])
        total_frames += _voiced_durations[i]
        if total_frames >= n_frames:
            voiced_durations[-1] -= total_frames - n_frames
            break
    return voiced_durations, silent_durations
