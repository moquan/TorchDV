# tests.py
# A file for keeping temporary tests; may re-use, maybe not

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy

from frontend_mw545.modules import make_logger, read_file_list, File_List_Selecter
from frontend_mw545.data_io import Data_File_IO


class Tests_Temp(object):
    """docstring for Tests_Temp"""
    def __init__(self, cfg=None):
        super(Tests_Temp, self).__init__()
        self.cfg = cfg
        self.logger = make_logger("Tests_Temp")

        self.DIO = Data_File_IO(self.cfg)

    def run(self):
        self.logger.info('running tests')
        self.run_4()

    def run_1(self):
        self.logger.info('Compare 2 directories')
        dir_1 = '/data/vectra2/tts/mw545/Data/Data_Voicebank_16kHz/label/nn_lab_resil'
        dir_2 = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/nn_lab_resil'

        l_1 = os.listdir(dir_1)
        l_2 = os.listdir(dir_2)
        # 1. compare file names; print exclusive ones
        #   result: 12 files only in /data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/nn_lab_resil
        #   ['p41_healthy_11.lab', 'p41_healthy_17.lab', 'p41_healthy_13.lab', 'p41_healthy_16.lab', 'p179_026.lab', 'p41_healthy_19.lab', 'p41_healthy_18.lab', 'p41_healthy_21.lab', 'p41_healthy_12.lab', 'p41_healthy_14.lab', 'p41_healthy_15.lab', 'p41_healthy_10.lab']
        #   No problem; p179 is a training speaker, p179_026 is not used
        if True:
            r_1 = []
            r_2 = []
            for x in l_1:
                if x not in l_2:
                    r_1.append(x)
            for x in l_2:
                if x not in l_1:
                    r_2.append(x)

            if len(r_1) > 0:
                self.logger.info('%i files only in %s' % (len(r_1), dir_1))
                print(r_1)

            if len(r_2) > 0:
                self.logger.info('%i files only in %s' % (len(r_2), dir_2))
                print(r_2)

        # 2. compare (a subset of) files, see if data are identical
        #   result: 
        #   different sizes, p202_037.lab; sizes: 633454, 573354
        #   different sizes, p202_162.lab; sizes: 556526, 496426
        #   p202_151.lab; sizes: 894288, 834188
        #   3 out of 100 has different sizes
        #   No problem; p202 is not used

        if True:
            num_files = 1000
            l_sub = numpy.random.choice(l_2, num_files, replace=False)

            len_list  = []
            data_list = []
            for x in l_sub:
                f_1 = os.path.join(dir_1, x)
                f_2 = os.path.join(dir_2, x)
                d_1, l_1 = self.DIO.load_data_file_frame(f_1, feat_dim=1)
                d_2, l_2 = self.DIO.load_data_file_frame(f_2, feat_dim=1)

                if l_1 != l_2:
                    self.logger.info('different sizes, %s; sizes: %i, %i' %(x, l_1, l_2))
                    len_list.append(x)
                else:
                    d = numpy.sum(d_1 - d_2)
                    if d != 0:
                        self.logger.info('different data, %s; sum_diff: %f' %(x, d))
                        data_list.append(x)

            if len(len_list) > 0:
                self.logger.info('%i out of %i has different sizes' % (len(len_list), num_files))
                print(len_list)

            if len(data_list) > 0:
                self.logger.info('%i out of %i has different data' % (len(data_list), num_files))
                print(data_list)

    def run_2(self):
        # Pytorch CNN test
        # Fake input data:
        #   4*100*86
        # CNN: 256D, kernel 40, stride 1
        #   CNN input is N*C*L, or N*D*T
        # expected output: 4*61*256
        # Result of print(y.shape):
        #   torch.Size([4, 256, 61])
        # Just need to transpose again
        import torch
        x = torch.zeros(4,100,86)
        c = torch.nn.Conv1d(86,256,40)
        y = c(torch.transpose(x, 1, 2))
        z = torch.transpose(z, 1, 2)
        print(y.shape)

    def run_3(self):
        # Pytorch pack_padded_sequence test
        #   dimensions: B*T*D if batch_first is True
        # Use a few sequences of different lengths, unsorted
        # Result:
        #   1. DO NOT USE unless necessary
        #   2. it's not a tensor
        #   3. it is specific for RNN-based models
        #   4. it can be generated by the model, as matrix and lengths are fed into
        import torch
        from torch.nn.utils.rnn import pack_padded_sequence
        from torch.nn.utils.rnn import pad_packed_sequence

        len_list = [5,4,6,10]
        D = 1
        B = len(len_list)
        T = max(len_list)

        data_1 = numpy.zeros((B,T,D))
        for b in range(B):
            data_1[b,:len_list[b],0] = range(len_list[b])
        # print(data_1)

        self.logger.info('tensor')
        data_2 = torch.tensor(data_1)
        print(data_2)
        pack_3 = pack_padded_sequence(data_2, len_list, batch_first=True, enforce_sorted=False)
        self.logger.info('pack_padded_sequence')
        print(pack_3)
        print(pack_3[0])
        print(isinstance(pack_3, (torch.Tensor,)))

    def run_4(self):
        import torch
        # simple tests...
        # test torch.mul, broadcasting
        # test mask
        S = 4
        B = 10
        D = 5
        out_lens = [3,10,5,1]

        x_SBD = torch.ones(S,B,D) * 1.5
        x_mask = numpy.zeros((S,B))
        for i,l in enumerate(out_lens):
            x_mask[i,:l] = 1.
        x_mask_tensor = torch.tensor(x_mask)
        out_lens_tensor = torch.tensor(out_lens)

        x_mask_SB1 = torch.unsqueeze(x_mask_tensor, 2)
        y = torch.mul(x_SBD, x_mask_SB1)
        y_SD_sum = torch.sum(y, dim=1, keepdim=False)
        out_lens_S1 = torch.unsqueeze(out_lens_tensor, 1)
        y_SD = torch.true_divide(y_SD_sum, out_lens_S1)
        print(y_SD)

        # test torch.nn.softmax, dim sum to 1

