#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Evaluate duration differences between generated and groundtruth audios."""

import argparse
import fnmatch
import logging
import multiprocessing as mp
import os

from typing import Dict
from typing import List
from typing import Tuple

import librosa
import numpy as np
import pysptk
import soundfile as sf

from fastdtw import fastdtw
from scipy import spatial


def find_files(
    root_dir: str, query: List[str] = ["*.flac", "*.wav"], include_root_dir: bool = True
) -> List[str]:
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (List[str]): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        List[str]: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for q in query:
            for filename in fnmatch.filter(filenames, q):
                files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def _get_basename(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


def calculate(
    file_list: List[str],
    gt_file_list: List[str],
    args: argparse.Namespace,
    output_dict: Dict,
):
    """Calculate MCD."""
    for i, gen_path in enumerate(file_list):
        corresponding_list = list(
            filter(lambda gt_path: _get_basename(gt_path) in gen_path, gt_file_list)
        )
        assert len(corresponding_list) == 1
        gt_path = corresponding_list[0]
        gt_basename = _get_basename(gt_path)

        # load wav file as int16
        gen_x, gen_fs = sf.read(gen_path, dtype="int16")
        gt_x, gt_fs = sf.read(gt_path, dtype="int16")

        fs = gen_fs
        if gen_fs != gt_fs:
            gt_x = librosa.resample(gt_x.astype(np.float), gt_fs, gen_fs)

        l_gen = gen_x.shape[0]
        l_gt  = gt_x.shape[0]

        d = np.abs((l_gen-l_gt)/l_gt)
        # logging.info(f"{gt_basename} {d:.4f}")
        output_dict[gt_basename] = d


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Duration differences.")
    parser.add_argument(
        "--wavdir",
        required=True,
        type=str,
        help="Path of directory for generated waveforms.",
    )
    parser.add_argument(
        "--gt_wavdir",
        required=True,
        type=str,
        help="Path of directory for ground truth waveforms.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Path of directory to write the results.",
    )

    # analysis related
    parser.add_argument(
        "--mcep_dim",
        default=None,
        type=int,
        help=(
            "Dimension of mel cepstrum coefficients. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--mcep_alpha",
        default=None,
        type=float,
        help=(
            "All pass constant for mel-cepstrum analysis. "
            "If None, automatically set to the best dimension for the sampling."
        ),
    )
    parser.add_argument(
        "--n_fft", default=1024, type=int, help="The number of FFT points."
    )
    parser.add_argument(
        "--n_shift", default=256, type=int, help="The number of shift points."
    )
    parser.add_argument(
        "--n_jobs", default=16, type=int, help="Number of parallel jobs."
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run MCD calculation in parallel."""
    args = get_parser().parse_args()

    # logging info
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # find files
    gen_files = sorted(find_files(args.wavdir))
    gt_files = sorted(find_files(args.gt_wavdir))

    # Get and divide list
    if len(gen_files) == 0:
        raise FileNotFoundError("Not found any generated audio files.")
    if len(gen_files) > len(gt_files):
        raise ValueError(
            "#groundtruth files are less than #generated files "
            f"(#gen={len(gen_files)} vs. #gt={len(gt_files)}). "
            "Please check the groundtruth directory."
        )
    logging.info("The number of utterances = %d" % len(gen_files))
    file_lists = np.array_split(gen_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        output_dict = manager.dict()
        processes = []
        for f in file_lists:
            p = mp.Process(target=calculate, args=(f, gt_files, args, output_dict))
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        # convert to standard list
        output_dict = dict(output_dict)

        # calculate statistics
        mean_dtw = np.mean(np.array([v for v in output_dict.values()]))
        std_dtw = np.std(np.array([v for v in output_dict.values()]))
        logging.info(f"Average: {mean_dtw:.4f} & {std_dtw:.4f} & ")

    # write results
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        with open(f"{args.outdir}/utt2durdiff", "w") as f:
            for utt_id in sorted(output_dict.keys()):
                dtw = output_dict[utt_id]
                f.write(f"{utt_id} {dtw:.4f}\n")
        with open(f"{args.outdir}/result_duration.txt", "w+") as f:
            f.write(f"#utterances: {len(gen_files)}\n")
            f.write(f"Average: {mean_dtw:.4f} & {std_dtw:.4f} & ")

    logging.info("Successfully finished duration difference evaluation.")


if __name__ == "__main__":
    main()
