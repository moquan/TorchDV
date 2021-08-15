# data_converter.py

import os, sys, pickle, time, shutil, copy
import math, numpy

from frontend_mw545.modules import make_logger, read_file_list
from frontend_mw545.data_io import Data_File_IO

class Data_File_Converter(object):
    """ 
    Data_File_Converter; 
    each method uses parameters from args or cfg 
    """
    def __init__(self, cfg=None):
        super(Data_File_Converter, self).__init__()
        self.logger = make_logger("DFC")

        self.cfg = cfg
        self.DIO = Data_File_IO(cfg)

        # Vocoder modules
        # Only import them when needed
        self.vocoder_analysis   = None
        self.vocoder_synthesize = None

    def wav_2_acoustic(self, in_file_name, out_file_dict, acoustic_in_dimension_dict=None, verbose_level=0):
        ''' Need to run in Python 2.7! '''
        ''' File based operation; wav file into feature files '''
        if self.vocoder_analysis is None:
            from pulsemodel.analysis import analysisf
            self.vocoder_analysis = analysisf
        if acoustic_in_dimension_dict is None:
            acoustic_in_dimension_dict = self.cfg.acoustic_in_dimension_dict

        self.vocoder_analysis(in_file_name,
            shift=0.005, dftlen=4096,
            finf0txt=None, f0_min=60, f0_max=600, ff0=out_file_dict['lf0'], f0_log=True, finf0bin=None,
            fspec=out_file_dict['mgc'], spec_mceporder=acoustic_in_dimension_dict['mgc']-1, spec_fwceporder=None, spec_nbfwbnds=None,
            fpdd=None, pdd_mceporder=None, fnm=out_file_dict['bap'], nm_nbfwbnds=acoustic_in_dimension_dict['bap'],
            verbose=verbose_level)

    def acoustic_2_wav(self, in_file_dict, out_file_name, synthesis_wav_sr=None, verbose_level=0):
        ''' Need to run in Python 2.7! '''
        if self.vocoder_synthesize is None:
            from pulsemodel.synthesis import synthesizef
            self.vocoder_synthesize = synthesizef
        if synthesis_wav_sr is None:
            synthesis_wav_sr = self.cfg.synthesis_wav_sr

        self.vocoder_synthesize(synthesis_wav_sr, shift=0.005, dftlen=4096, 
            ff0=None, flf0=in_file_dict['lf0'], 
            fspec=None, ffwlspec=None, ffwcep=None, fmcep=in_file_dict['mgc'], 
            fnm=None, ffwnm=in_file_dict['bap'], nm_cont=False, fpdd=None, fmpdd=None, 
            fsyn=out_file_name, verbose=verbose_level)

    def pml_2_cmp(self, in_file_dict, out_file_name):
        ''' 
        stack to form cmp
        Note: this is a simplified version
            Simple stacking
            Does not use MLPG for deltas
        '''
        cfg = self.cfg
        data_dict = {}
        frame_number_min = numpy.inf

        for feat_name in cfg.acoustic_features:
            feat_dim  = cfg.acoustic_in_dimension_dict[feat_name]
            feat_data, frame_number = self.DIO.load_data_file_frame(in_file_dict[feat_name], feat_dim)
            data_dict[feat_name] = feat_data
            if frame_number < frame_number_min:
                frame_number_min = frame_number

        cmp_data = numpy.zeros((frame_number_min, cfg.nn_feature_dims['cmp']))
        start_index = 0
        for feat_name in cfg.acoustic_features:
            feat_dim  = cfg.acoustic_in_dimension_dict[feat_name]
            cmp_data[:, start_index:start_index+feat_dim] = data_dict[feat_name][:frame_number_min, :]
            start_index += feat_dim

        self.DIO.save_data_file(cmp_data, out_file_name)

    def cmp_2_pml(self, in_file_name, out_file_dict):
        ''' split to individual vocoder file '''
        cfg = self.cfg

        cmp_data, frame_number = self.DIO.load_data_file_frame(in_file_name, cfg.nn_feature_dims['cmp'])

        data_dict = {}
        start_index = 0
        for feat_name in cfg.acoustic_features:
            feat_dim  = cfg.acoustic_in_dimension_dict[feat_name]
            data_dict[feat_name] = cmp_data[:, start_index:start_index+feat_dim]
            self.DIO.save_data_file(data_dict[feat_name], out_file_dict[feat_name])
            start_index += feat_dim

    def wav_2_wav_cmp(self, in_file_name, out_file_name, cmp_rate=200):
        ''' 
        Strip waveform header first
        Make "cmp" style file, by reshaping waveform
        Discard residuals to make "frames"; for silence removal later
        '''
        # find frame number, remove residual to make whole frames, quantise
        data, sr = self.DIO.read_wav_2_wav_1D_data(in_file_name, return_sample_rate=True)
        dim = sr / cmp_rate
        assert len(data.shape) == 1
        num_frames = int(data.shape[0] / dim)
        # remove residual samples i.e. less than a frame
        num_samples = int(dim * num_frames)
        new_data = numpy.array(data[:num_samples], dtype='float32')
        self.DIO.save_data_file(new_data, out_file_name)
        return sr

    def wav_cmp_2_wav(self, in_file_name, out_file_name, sr=16000):
        wav_1D_data, num_frames = self.DIO.load_data_file_frame(in_file_name, 1)
        wav_1D_data = numpy.array(cmp_data, dtype='int16')
        self.DIO.write_wav_1D_data_2_wav(wav_1D_data, out_file_name, sample_rate=sr, cfg=self.cfg)

    def wav_cmp_2_wav_mu_law_cmp(self, in_file_name, out_file_name, mu_value=255.):
        '''
        apply mu-law (ITU-T, 1988)
        '''
        ori_data, num_frames = self.DIO.load_data_file_frame(in_file_name, 1)
        mu_data = numpy.sign(ori_data) * numpy.log(1.+mu_value*numpy.abs(ori_data)) / numpy.log(1.+mu_value)
        self.DIO.save_data_file(mu_data, out_file_name)

    def wav_mu_law_cmp_2_wav_cmp(self, in_file_name, out_file_name, mu_value=255.):
        '''
        Revert mu-law (ITU-T, 1988)
        '''
        mu_data, num_frames = self.DIO.load_data_file_frame(in_file_name, 1)
        ori_data = numpy.sign(mu_data) * (1./mu_value) * ( numpy.power((1.+mu_value), numpy.abs(mu_data)) - 1.)
        self.DIO.save_data_file(ori_data, out_file_name)

    def pitch_text_2_pitch_1D(self, in_file_name, wav_cmp_file_name, out_file_name):
        ''' 
        Make pitch file:
            same length as the wav_cmp file
            (next pitch time) - (start of current sample time)
            pitch = 0: pitch is at start of current sample window
        Get sample rate from cfg
        After last pitch location: pitch time = -1.
        '''
        sr = self.cfg.wav_sr

        pitch_list = self.DIO.read_pitch_reaper(in_file_name)
        wav_data, num_samples = self.DIO.load_data_file_frame(wav_cmp_file_name, 1)

        pitch_data = numpy.arange(num_samples) / float(sr) * (-1.)
        n_start = 0
        for pitch_t in pitch_list:
            n_end = int(pitch_t*sr) # This index also starts before the pitch; careful with +-1
            pitch_data[n_start:n_end+1] += pitch_t
            n_start = n_end+1

        # if (n_start) < num_samples:
        #     pitch_data[n_start:num_samples] = -1.
        pitch_data[pitch_data<0] = -1.

        self.DIO.save_data_file(pitch_data, out_file_name)

    def lf0_2_lf0_16kHz(self, in_file_name, out_file_name):
        '''
        Make lf0 16kHz file:
            Load lf0 file, which is 200Hz
            Up-sample to 16kHz; Linear interpolate in between, Copy at the ends
        '''
        wav_sr = self.cfg.wav_sr
        frame_sr = self.cfg.frame_sr
        sr_ratio = int(wav_sr / frame_sr)
        
        lf0_data, num_frames = self.DIO.load_data_file_frame(in_file_name, 1)
        num_samples = num_frames * sr_ratio

        lf016k_data = numpy.zeros(num_samples)
        window_size = sr_ratio
        half_window_size = int(window_size /2)
        n_inc = numpy.arange(window_size) / float(window_size-1)
        n_dec = 1. - n_inc
        # Copy for both ends
        lf016k_data[:half_window_size] = lf0_data[0]
        lf016k_data[num_samples-half_window_size:] = lf0_data[-1]

        for n in range(num_frames-1):
            n_start = half_window_size+n*window_size
            n_end   = n_start + window_size # Exclusive
            x = lf0_data[n]
            y = lf0_data[n+1]

            lf016k_data[n_start:n_end] = x * n_dec + y * n_inc
        self.DIO.save_data_file(lf016k_data, out_file_name)

    def lf0_2_f0(self, in_file_name, out_file_name, remove_lf0_file=False):
        lf0_data, num_frames = self.DIO.load_data_file_frame(in_file_name, 1)
        f0_data = numpy.exp(lf0_data)
        self.DIO.save_data_file(f0_data, out_file_name)
        if remove_lf0_file:
            os.remove(in_file_name)

    def f0_reaper_2_f0_200Hz(self, in_file_name, out_file_name, remove_in_file=False):
        '''
        Read REAPER-stype f0 file
        Linear-interpolate the unvoiced regions; copy first/last frame for the ends
        Save to data file
        '''
        f0_reaper_data = self.DIO.read_f0_reaper(in_file_name)
        file_len = len(f0_reaper_data)

        # Interpolation method; (Gilles)
        # Stack to make two column matrix [time[s], value[Hz]] (ljuvela)
        ts = (0.005)*numpy.arange(file_len)
        f0s = numpy.vstack((ts, f0_reaper_data)).T

        f0s[:,1] = numpy.interp(f0s[:,0], f0s[f0s[:,1]>0,0], f0s[f0s[:,1]>0,1])

        self.DIO.save_data_file(f0s[:,1], out_file_name)

    def f0_200Hz_2_f0_16kHz(self, in_file_name, out_file_name, remove_in_file=False):
        '''
        Upsample from 200Hz to 16kHz
        Linear interpolation in between
        '''
        f0_200Hz_data, l = self.DIO.load_data_file_frame(in_file_name, 1)
        x_200Hz = (0.005)*numpy.arange(l) + (1./400.)
        x_16kHz = (0.005/80.)*numpy.arange(l*80) + (1./32000.)

        f0_16kHz_data = numpy.interp(x_16kHz, x_200Hz, f0_200Hz_data[:,0])

        self.DIO.save_data_file(f0_16kHz_data, out_file_name)

class Data_File_List_Converter(object):
    """ Data File List Converter; processes lists of data files """
    def __init__(self, cfg=None):
        super(Data_File_List_Converter, self).__init__()
        self.logger = make_logger("DFC_List")

        self.cfg = cfg
        self.DFC = Data_File_Converter(cfg)
        
        file_id_list_file = cfg.file_id_list_file['used']
        # file_id_list_file = cfg.file_id_dv_test_list_file
        
        self.logger.info('Reading file list from %s' % file_id_list_file)
        self.file_id_list = read_file_list(file_id_list_file)

    def wav_2_acoustic_list(self):
        ''' Generate vocoder files '''
        cfg = self.cfg
        
        vocoder_file_dict = {}
        for file_id in self.file_id_list:
            wav_file = os.path.join(cfg.wav_dir, file_id + '.wav')
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            self.logger.info('Generating for file %s' % wav_file)
            self.DFC.wav_2_acoustic(wav_file, vocoder_file_dict)

    def pitch_text_2_pitch_1D_list(self):
        ''' Generate pitch 1D files '''
        cfg = self.cfg

        for file_id in self.file_id_list:
            pitch_reaper_file = os.path.join(cfg.pitch_dir, file_id + '.pitch')
            wav_cmp_file    = os.path.join(cfg.nn_feat_dirs['wav'], file_id + '.wav')
            pitch_1D_file   = os.path.join(cfg.nn_feat_dirs['pitch'], file_id + '.pitch')

            self.logger.info('Saving to file %s' % pitch_1D_file)
            self.DFC.pitch_text_2_pitch_1D(pitch_reaper_file, wav_cmp_file, pitch_1D_file)

    def lf0_2_lf0_16kHz_list(self):
        ''' Generate pitch 1D files '''
        cfg = self.cfg
        lf0_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz_WAV_16kHz/PML/lf0'
        lf016k_dir = '/data/vectra2/tts/mw545/Data/data_voicebank/nn_lf016k'

        for file_id in self.file_id_list:
            lf0_file = os.path.join(lf0_dir, file_id + '.lf0')
            lf016k_file = os.path.join(lf016k_dir, file_id + '.lf016k')

            self.logger.info('Saving to file %s' % lf016k_file)
            self.DFC.lf0_2_lf0_16kHz(lf0_file, lf016k_file)

    def pml_2_cmp_list(self):
        ''' Generate vocoder files '''
        cfg = self.cfg

        for file_id in self.file_id_list:
            cmp_file = os.path.join(cfg.nn_feat_dirs['cmp'], file_id + '.cmp')
            vocoder_file_dict = {}
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            self.logger.info('Generating file %s' % cmp_file)
            self.DFC.pml_2_cmp(vocoder_file_dict, cmp_file)

    def lf0_2_f0_list(self):
        ''' Convert nn_lf016k_resil files to nn_f016k_resil '''
        cfg = self.cfg

        lf0_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_lf016k_resil'
        f0_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f016k_resil'

        for file_id in self.file_id_list:
            lf016k_file = os.path.join(lf0_dir, file_id + '.lf016k')
            f016k_file  = os.path.join(f0_dir, file_id + '.f016k')

            self.logger.info('Saving to file %s' % f016k_file)
            self.DFC.lf0_2_f0(lf016k_file, f016k_file, remove_lf0_file=True)

    def f0_reaper_2_f0_200Hz_list(self):
        ''' Read reaper f0 file, interpolate unvoiced regions '''
        cfg = self.cfg

        f0_reaper_dir = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/reaper_16kHz/f0'
        f0_200Hz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f0200'

        for file_id in self.file_id_list:
            f0_reaper_file = os.path.join(f0_reaper_dir, file_id + '.f0')
            f0_200Hz_file  = os.path.join(f0_200Hz_dir, file_id + '.f0200')

            self.logger.info('Saving to file %s' % f0_200Hz_file)
            self.DFC.f0_reaper_2_f0_200Hz(f0_reaper_file, f0_200Hz_file)

    def f0_200Hz_list_2_f0_16kHz_list(self):
        ''' Read f0 file, upsample from 200Hz to 16kHz '''
        cfg = self.cfg

        f0_200Hz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f0200'
        f0_16kHz_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/data/nn_f016k'

        for file_id in self.file_id_list:
            f0_200Hz_file  = os.path.join(f0_200Hz_dir, file_id + '.f0200')
            f0_16kHz_file  = os.path.join(f0_16kHz_dir, file_id + '.f016k')

            self.logger.info('Saving to file %s' % f0_16kHz_file)
            self.DFC.f0_200Hz_2_f0_16kHz(f0_200Hz_file, f0_16kHz_file)

#################################################
#     Write Bash Scripts to process files       #
#   Submit to grid, one script/job per speaker  #
#################################################
def reaper_all(cfg):
    '''
    Use Reaper to extract f0 and pitch
    Write a bash script for each speaker
    Write a submit script to qsub all speaker scripts
    '''

    bash_script_dir  = '/home/dawna/tts/mw545/TorchDV/debug_nausicaa/bash_script'
    input_wav_dir    = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/wav24'
    output_f0_dir    = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/reaper_24kHz/f0'
    output_pitch_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/reaper_24kHz/pitch'

    file_id_list_file = cfg.file_id_list_file['used']
    file_id_list = read_file_list(file_id_list_file)
    
    speaker_id_list = cfg.speaker_id_list_dict['all']

    # Write speaker bash; one bash file per speaker
    if True:
        f_speaker = {}
        for speaker_id in speaker_id_list:
            f_speaker_file_name = os.path.join(bash_script_dir, speaker_id+'.sh')
            f_speaker[speaker_id] = open(f_speaker_file_name, 'w')

        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            l = 'reaper -i %s/%s/%s.wav -f %s/%s/%s.f0 -p %s/%s/%s.pitch -a \n' % (input_wav_dir, speaker_id, file_id, output_f0_dir, speaker_id, file_id, output_pitch_dir, speaker_id, file_id)
            f_speaker[speaker_id].write(l)

        for speaker_id in speaker_id_list:
            f_speaker_file_name = os.path.join(bash_script_dir, speaker_id+'.sh')
            f_speaker[speaker_id].close()

    # Write submit script
    if True:
        f_submit_file_name = os.path.join(bash_script_dir, 'submit.sh')
        f_submit_file = open(f_submit_file_name, 'w')

        for speaker_id in speaker_id_list:
            l = 'qsub -S /bin/bash  -o ${PWD} -e ${PWD} -l queue_priority=low,tests=0,mem_grab=0M,osrel=* %s.sh \n' % speaker_id
            f_submit_file.write(l)

        f_submit_file.close()


#########################
# Main function to call #
#########################

def run_Data_File_List_Converter(cfg):
    DFC_List = Data_File_List_Converter(cfg)
    DFC_List.f0_200Hz_list_2_f0_16kHz_list()