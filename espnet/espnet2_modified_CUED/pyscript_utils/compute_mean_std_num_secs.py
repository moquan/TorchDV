#!/usr/bin/env python3

import argparse
import logging
import os
import numpy

from typing import Dict
from typing import List
from typing import Tuple

class Compute_Loss_Mean_Std(object):
    """
    Compute the mean and std of a loss from 30 draws
    """
    def __init__(self):
        super(Compute_Loss_Mean_Std, self).__init__()
        self.loss_name_list = ['l1_loss', 'mse_loss', 'bce_loss', 'attn_loss', 'loss']

    def run(self, log_dir, num_secs, loss_to_compute='mse_loss'):
        full_dir_name = os.path.join(log_dir, 'same_%s_seconds_per_speaker_draw_[n2]/eval1/log' % num_secs)
        mse_list_i_file = []
        for j in range(1,31):
            log_dir = full_dir_name.replace('[n2]', str(j))
            total_loss_dict = self.get_losses_from_dir(log_dir, upper_limit=10)
            # mse_list_i_file.append(numpy.mean(total_loss_dict['l1_loss']))
            mse_list_i_file.append(numpy.mean(total_loss_dict[loss_to_compute]))
        m = numpy.mean(mse_list_i_file)
        # mse_std_list.append(numpy.mean(numpy.std(mse_list_i_file,axis=0,ddof=1)))
        s = numpy.std(mse_list_i_file,ddof=1)
        return m,s

    def format_loss_dict(self, loss_dict):
        # format loss dict for better printing and fill in form
        # 4 d.p., separated by '&'
        # except attention_loss, very small, use 4 s.f.
        s = ''
        for k in self.loss_name_list:
            if k == 'attn_loss':
                s += '%.4E & ' % loss_dict[k]
            else:
                s += '%.4f & ' % loss_dict[k]
        s = s[:-2]
        s += '\\\\'
        return s

    def read_log_file(self, full_file_name):
        # return a dict of lists of loss, and number of files
        with open(full_file_name, 'r') as f:
            file_lines = f.readlines()
        file_loss_dict = {k:[] for k in self.loss_name_list}
        num_files = 0.
        for line in file_lines:
            # {'l1_loss': 0.5132738947868347, 'mse_loss': 0.22509634494781494, 'bce_loss': 0.010223600082099438, 'attn_loss': 3.832143556792289e-05, 'loss': 0.7486321926116943}
            if line[0] == '{':  # dict of losses, start with {
                num_files += 1
                k_v_pairs = line.strip().split(', ')
                for k_v_p in k_v_pairs:
                    k = k_v_p.split(':')[0].split("'")[1]
                    v = k_v_p.split(' ')[1]
                    if v[-1] == '}':
                        v = v[:-1]
                    v = float(v)
                    file_loss_dict[k].append(v)

        return file_loss_dict, num_files

    def get_losses_from_dir(self, dir_name, upper_limit=100):
        # extract loss from all log files tts_inference.*.log
        total_loss_dict = {k:[] for k in self.loss_name_list}
        total_num_file  = 0.

        file_list = os.listdir(dir_name)
        for file_name in file_list:
            if file_name.split('.')[0] == 'tts_inference' and file_name.split('.')[-1] == 'log':
                if int(file_name.split('.')[1]) <= upper_limit:
                    full_file_name = os.path.join(dir_name, file_name)
                    file_loss_dict, num_files = self.read_log_file(full_file_name)
                    for k in self.loss_name_list:
                        total_loss_dict[k].extend(file_loss_dict[k])
                        total_num_file += num_files

        if total_num_file == 0:
            print('0 files found in %s!' % dir_name)
        # return {k:total_loss_dict[k]/total_num_file for k in self.loss_name_list}
        return total_loss_dict

    def get_loss_dict(self, exp_dir_name):
        loss_list_dict = {k:[] for k in self.loss_name_list}
        for i in range(30):
            dir_name = os.path.join(exp_dir_name, 'tf_%i/eval1/log' % (i+1))
            loss_dict = self.get_losses_from_dir(dir_name)

            for k in self.loss_name_list:
                loss_list_dict[k].append(loss_dict[k])
        return loss_list_dict


def get_parser() -> argparse.Namespace:
    """Get argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Mel-cepstrum distortion.")
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path of directory of log files.",
    )
    parser.add_argument(
        "num_secs",
        type=str,
        help="Number of seconds to compute.",
    )
    parser.add_argument(
        "--loss_to_compute",
        default='mse_loss',
        type=str,
        help="Name of loss to compute",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Path of directory to write the results.",
    )
    parser.add_argument(
        "--verbose",
        default=1,
        type=int,
        help="Verbosity level. Higher is more logging.",
    )
    return parser


def main():
    """Run log-F0 RMSE calculation in parallel."""
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

    c = Compute_Loss_Mean_Std()
    m,s = c.run(args.log_dir, args.num_secs, args.loss_to_compute)
    logging.info('Results of %s, %ss from %s' %(args.loss_to_compute, args.num_secs, args.log_dir))
    logging.info(f"Average: {m:.4f} +- {s:.4f}")


if __name__ == "__main__":
    main()
