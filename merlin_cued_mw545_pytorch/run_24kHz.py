import os, sys, pickle, time, shutil, logging
import math, numpy, scipy, scipy.io.wavfile#, sigproc, sigproc.pystraight

from frontend_mw545.modules import make_logger, read_file_list, prepare_script_file_path, log_class_attri

class prep_24kHz_dir(object):
    def __init__(self, cfg):
        '''
        Things to do:
        1. Re-write all file_id_lists files; pad speaker id with 0s. Sort too.
        2. Link wav24 properly
          Use new style: split by speaker
          Need to modify a lot of data_processing files, be careful
        3. Make vocoder, cmp, norm, resil
        '''
        self.logger = make_logger("prep_24kHz")

        self.cfg = cfg

        file_id_list_file = cfg.file_id_list_file['used']
        self.file_id_list = read_file_list(file_id_list_file)

        from frontend_mw545.data_converter import Data_File_Converter
        self.DFC = Data_File_Converter(cfg)

    def run(self):
        # self.rewrite_file_id_lists()
        # self.rewrite_file_id_lists_meta()
        # self.make_pml(debug_mode=False)
        # self.make_f0_pitch()
        # self.make_cmp()
        # self.make_wav_cmp()
        # self.make_pitch_1D()
        # self.make_f0_1D()
        # self.make_new_speaker_padded_dirs('label_state_align')
        # self.make_new_speaker_padded_dirs('nn_lab', debug_mode=False)
        # self.make_new_speaker_padded_dirs('nn_lab_resil', debug_mode=False)
        # self.make_new_speaker_padded_dirs('nn_lab_resil_norm_601', debug_mode=False)

        # self.reduce_silence('cmp')
        # self.cmp_norm_files()

        # self.reduce_silence('f024k', last_speaker_id=0)
        # self.reduce_silence('pitch', last_speaker_id=0)
        # self.reduce_silence('wav', last_speaker_id=0)
        # self.wav_norm_files(last_speaker_id=0)

        # self.make_cmp_shift_data()
        
        self.run_tests()

    def make_cmp_shift_data(self, num_samples_shift=50):
        self.logger.info('make_cmp_shift_data')
        cfg = self.cfg

        from frontend_mw545.data_io import Data_File_IO
        self.DIO = Data_File_IO(cfg)
        shift_data_dir = '/data/mifs_scratch/mjfg/mw545/dv_pos_test'
        file_id_list = read_file_list(self.cfg.file_id_list_file['dv_pos_test'])

        # 1. make wav_shift
        if False:
            self.logger.info('make wav_shift')
            in_dir_name  = cfg.wav_dir
            out_dir_name = os.path.join(shift_data_dir, 'wav_shift')
            wav_sr = cfg.wav_sr
            self.prepare_speaker_dir(out_dir_name)

            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                in_file_name  = os.path.join(in_dir_name, speaker_id, file_id+'.wav')
                out_file_name = os.path.join(out_dir_name, speaker_id, file_id+'.wav')
                wav_data_orig = self.DIO.read_wav_2_wav_1D_data(in_file_name)

                for i in range(num_samples_shift+1):
                    wav_data_shifted = wav_data_orig[i:]
                    out_file_name_shifted = out_file_name + '.%i'%i
                    self.logger.info('Save to %s' % out_file_name_shifted)
                    self.DIO.write_wav_1D_data_2_wav(wav_data_shifted, out_file_name_shifted, wav_sr)

        # 2. make PML_shift
        if False:
            self.logger.info('make PML_shift')
            in_dir_name  = os.path.join(shift_data_dir, 'wav_shift')
            out_dir_name = os.path.join(shift_data_dir, 'PML_shift')
            for feat_name in cfg.acoustic_features:
                self.prepare_speaker_dir(os.path.join(out_dir_name, feat_name))

            wav_file_list = []
            vocoder_file_dict_list = []
            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                for i in range(num_samples_shift+1):
                    vocoder_file_dict = {}
                    wav_file = os.path.join(in_dir_name, speaker_id, file_id + '.wav'+ '.%i'%i)
                    for feat_name in cfg.acoustic_features:
                        vocoder_file_dict[feat_name] = os.path.join(out_dir_name, feat_name, speaker_id, file_id + cfg.acoustic_file_ext_dict[feat_name]+ '.%i'%i)
                    wav_file_list.append(wav_file)
                    vocoder_file_dict_list.append(vocoder_file_dict)

            from multiprocessing import Pool
            from frontend_mw545.data_converter import Data_File_Converter
            global Data_File_Converter
            p = Pool(30)
            p.map(make_pml_single, zip(wav_file_list, vocoder_file_dict_list))
        
        # 3. make cmp_shift
        if False:
            self.logger.info('make cmp_shift')
            in_dir_name  = os.path.join(shift_data_dir, 'PML_shift')
            out_dir_name = os.path.join(shift_data_dir, 'cmp_shift')
            self.prepare_speaker_dir(out_dir_name)

            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                for i in range(num_samples_shift+1):
                    vocoder_file_dict = {}
                    cmp_file = os.path.join(out_dir_name, speaker_id, file_id + '.cmp'+ '.%i'%i)
                    for feat_name in cfg.acoustic_features:
                        vocoder_file_dict[feat_name] = os.path.join(in_dir_name, feat_name, speaker_id, file_id + cfg.acoustic_file_ext_dict[feat_name]+ '.%i'%i)
                    self.DFC.pml_2_cmp(vocoder_file_dict, cmp_file)

        # 4. cmp resil and norm
        if False:
            self.logger.info('make cmp_shift_resil')
            from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
            self.DSR = Data_Silence_Reducer(self.cfg)
            in_dir_name  = os.path.join(shift_data_dir, 'cmp_shift')
            out_dir_name = os.path.join(shift_data_dir, 'cmp_shift_resil')
            self.prepare_speaker_dir(out_dir_name)

            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                alignment_file_name = os.path.join(cfg.lab_dir, speaker_id, file_id + '.lab')
                for i in range(num_samples_shift+1):
                    in_file_name  = os.path.join(in_dir_name, speaker_id, file_id + '.cmp'+ '.%i'%i)
                    out_file_name = os.path.join(out_dir_name, speaker_id, file_id + '.cmp'+ '.%i'%i)

                    self.logger.info('Saving to file %s' % out_file_name)
                    self.DSR.reduce_silence_file(alignment_file_name, in_file_name, out_file_name, feat_name='cmp')

        if True:
            self.logger.info('make cmp_shift_resil_norm')
            from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
            self.DMVN = Data_Mean_Var_Normaliser(self.cfg)
            self.DMVN.load_mean_std_values()
            in_dir_name  = os.path.join(shift_data_dir, 'cmp_shift_resil')
            out_dir_name = os.path.join(shift_data_dir, 'cmp_shift_resil_norm')
            self.prepare_speaker_dir(out_dir_name)

            for file_id in file_id_list:
                speaker_id = file_id.split('_')[0]
                for i in range(num_samples_shift+1):
                    in_file_name  = os.path.join(in_dir_name, speaker_id, file_id + '.cmp'+ '.%i'%i)
                    out_file_name = os.path.join(out_dir_name, speaker_id, file_id + '.cmp'+ '.%i'%i)

                    self.logger.info('Saving to file %s' % out_file_name)
                    self.DMVN.norm_file(in_file_name, out_file_name)

    def run_tests(self):
        from frontend_mw545.tests import Tests_Temp
        tests = Tests_Temp(self.cfg)
        tests.run()

    def pad_speaker_id(self, file_id, l_id=4):
        # Pad speaker id with 0s, so lengths are all l_id
        speaker_id = file_id.split('_')[0]
        l = len(speaker_id)
        if l < l_id:
            # Pad zeros
            file_id_new = 'p' + (l_id-l)*'0' + file_id[1:]
            return file_id_new
        else:
            return file_id

    def unpad_speaker_id(self, file_id):
        # Remove the padded 0s in speaker id
        new_file_id = file_id
        while new_file_id[1] == '0':
            new_file_id = new_file_id[0] + new_file_id[2:]
        return new_file_id

    def prepare_speaker_dir(self, dir_name, speaker_id_list=None):
        '''
        Make directory
        Make speaker-level directories within
        '''
        if speaker_id_list is None:
            speaker_id_list = self.cfg.speaker_id_list_dict['all']

        prepare_script_file_path(dir_name)
        for speaker_id in speaker_id_list:
            prepare_script_file_path(os.path.join(dir_name, speaker_id))

    def rewrite_file_id_lists(self):
        new_file_id_list_dir  = cfg.file_id_list_dir
        prepare_script_file_path(new_file_id_list_dir)

        prev_file_id_list_dir = '/home/dawna/tts/mw545/TorchDV/file_id_lists'

        for k in cfg.file_id_list_file:
            new_file_name  = cfg.file_id_list_file[k]
            prev_file_name = new_file_name.replace(new_file_id_list_dir, prev_file_id_list_dir)

            prev_file_id_list = read_file_list(prev_file_name)
            new_file_id_list  = [self.pad_speaker_id(x) for x in prev_file_id_list]
            new_file_id_list.sort()

            print('write to %s' % new_file_name)
            with open(new_file_name,'w') as f:
                for file_id in new_file_id_list:
                    f.write(file_id+'\n')

    def rewrite_file_id_lists_meta(self):
        new_file_id_list_dir  = os.path.join(cfg.file_id_list_dir, 'data_meta')
        prepare_script_file_path(new_file_id_list_dir)

        prev_file_id_list_dir = os.path.join('/home/dawna/tts/mw545/TorchDV/file_id_lists', 'data_meta')

        file_name_list = ['file_id_list_num_extra_cmp_pos_test.scp', 'file_id_list_num_extra_wav_pos_test.scp', 'file_id_list_num_sil_frame.scp']

        for f in file_name_list:
            new_file_name  = os.path.join(new_file_id_list_dir, f)
            prev_file_name = os.path.join(prev_file_id_list_dir, f)

            prev_file_id_list = read_file_list(prev_file_name)
            new_file_id_list  = [self.pad_speaker_id(x) for x in prev_file_id_list]
            new_file_id_list.sort()

            print('write to %s' % new_file_name)
            with open(new_file_name,'w') as f:
                for file_id in new_file_id_list:
                    f.write(file_id+'\n')

    def make_new_speaker_padded_dir(self, out_dir, in_file_format, out_file_format, debug_mode=False):
        # Original dir: [in_dir]/p1_001.ext
        # New dir: [out_dir]/p001/p001_001.ext
        self.logger.info('make_new_speaker_padded_dir')
        self.logger.info('out_dir: %s' % out_dir)

        prepare_script_file_path(out_dir)
        self.prepare_speaker_dir(out_dir)

        if debug_mode:
            file_id_list = self.file_id_list[:10]
        else:
            file_id_list = self.file_id_list

        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            unpadded_speaker_id = self.unpad_speaker_id(speaker_id)
            unpadded_file_id = self.unpad_speaker_id(file_id)

            src_full_name = in_file_format.replace('[unpadded_speaker_id]', unpadded_speaker_id).replace('[unpadded_file_id]', unpadded_file_id)
            tar_full_name = out_file_format.replace('[speaker_id]', speaker_id).replace('[file_id]', file_id)
            os.symlink(src_full_name, tar_full_name)

    def make_new_speaker_padded_dirs(self, feat_name='label_state_align', debug_mode=False):
        if feat_name == 'label_state_align':
            out_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/label_state_align'
            in_file_format  = '/home/dawna/tts/data/VoiceBank48kHz/smaplr/[unpadded_speaker_id]/align/[unpadded_file_id].lab'
            out_file_format = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/label_state_align/[speaker_id]/[file_id].lab'
            self.logger.info('Attention! Check p320 and link/copy p320b')

        if feat_name == 'nn_lab':
            out_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab'
            in_file_format  = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/nn_lab/[unpadded_file_id].lab'
            out_file_format = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab/[speaker_id]/[file_id].lab'

        if feat_name == 'nn_lab_resil':
            out_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab_resil'
            in_file_format  = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/nn_lab_resil/[unpadded_file_id].lab'
            out_file_format = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab_resil/[speaker_id]/[file_id].lab'

        if feat_name == 'nn_lab_resil_norm_601':
            out_dir = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab_resil_norm_601'
            in_file_format  = '/data/vectra2/tts/mw545/Data/Data_Voicebank_48kHz/label/nn_lab_resil_norm_601/[unpadded_file_id].lab'
            out_file_format = '/data/vectra2/tts/mw545/Data/Data_Voicebank_24kHz/label/nn_lab_resil_norm_601/[speaker_id]/[file_id].lab'

        self.make_new_speaker_padded_dir(out_dir, in_file_format, out_file_format, debug_mode=debug_mode)

    def make_pml(self, debug_mode=False):
        # 1. Make all directories
        #   make speaker-level directories
        # 2. Generate pml files
        #   Use multiprocessing to speed-up
        #   Need to write an isolated function
        cfg = self.cfg
        for feat_name in cfg.acoustic_features:
            self.prepare_speaker_dir(cfg.acoustic_dir_dict[feat_name])

        if debug_mode:
            file_id_list = self.file_id_list[:10]
        else:
            file_id_list = self.file_id_list

        wav_file_list = []
        vocoder_file_dict_list = []
        for file_id in file_id_list:
            vocoder_file_dict = {}
            speaker_id = file_id.split('_')[0]
            wav_file = os.path.join(cfg.wav_dir, speaker_id, file_id + '.wav')
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], speaker_id, file_id + cfg.acoustic_file_ext_dict[feat_name])
            wav_file_list.append(wav_file)
            vocoder_file_dict_list.append(vocoder_file_dict)

        from multiprocessing import Pool
        from frontend_mw545.data_converter import Data_File_Converter
        global Data_File_Converter
        p = Pool(30)
        p.map(make_pml_single, zip(wav_file_list, vocoder_file_dict_list))

    def make_f0_pitch(self):
        cfg = self.cfg
        self.prepare_speaker_dir(cfg.reaper_pitch_dir)
        self.prepare_speaker_dir(cfg.reaper_f0_dir)

        from frontend_mw545.data_converter import reaper_all
        reaper_all(self.cfg)

    def make_cmp(self):
        # Make acoustic cmp from vocoder features
        self.logger.info('make_cmp')
        cfg = self.cfg
        file_id_list = self.file_id_list

        self.prepare_speaker_dir(cfg.nn_feat_dirs['cmp'])

        for file_id in file_id_list:
            vocoder_file_dict = {}
            speaker_id = file_id.split('_')[0]
            cmp_file = os.path.join(cfg.nn_feat_dirs['cmp'], speaker_id, file_id + '.cmp')
            for feat_name in cfg.acoustic_features:
                vocoder_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], speaker_id, file_id + cfg.acoustic_file_ext_dict[feat_name])
            self.DFC.pml_2_cmp(vocoder_file_dict, cmp_file)

    def make_wav_cmp(self):
        # Make wav cmp from waveform
        # Make vector-like sequence for silence reduction
        self.logger.info('make_wav_cmp')
        cfg = self.cfg
        file_id_list = self.file_id_list

        self.prepare_speaker_dir(cfg.nn_feat_dirs['wav'])

        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            wav_file = os.path.join(cfg.wav_dir, speaker_id, file_id + '.wav')
            wav_cmp_file = os.path.join(cfg.nn_feat_dirs['wav'], speaker_id, file_id + '.wav')
            self.DFC.wav_2_wav_cmp(wav_file, wav_cmp_file)

    def make_pitch_1D(self):
        # Make 1D pitch data
        self.logger.info('make_pitch_1D')
        cfg = self.cfg
        file_id_list = self.file_id_list

        self.prepare_speaker_dir(cfg.nn_feat_dirs['pitch'])
        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            pitch_reaper_file = os.path.join(cfg.reaper_pitch_dir, speaker_id, file_id + '.pitch')
            wav_cmp_file  = os.path.join(cfg.nn_feat_dirs['wav'], speaker_id, file_id + '.wav')
            pitch_1D_file = os.path.join(cfg.nn_feat_dirs['pitch'], speaker_id, file_id + '.pitch')

            self.DFC.pitch_text_2_pitch_1D(pitch_reaper_file, wav_cmp_file, pitch_1D_file)

    def make_f0_1D(self):
        # Make 1D f0 data
        self.logger.info('make_f0_1D')
        cfg = self.cfg
        file_id_list = self.file_id_list

        f0_200Hz_dir  = '/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz/nn_f0200'
        self.prepare_speaker_dir(f0_200Hz_dir)
        self.prepare_speaker_dir(cfg.nn_feat_dirs['f024k'])
        for file_id in file_id_list:
            speaker_id = file_id.split('_')[0]
            f0_reaper_file = os.path.join(cfg.reaper_f0_dir, speaker_id, file_id + '.f0')
            f0_200Hz_file  = os.path.join(f0_200Hz_dir, speaker_id, file_id + '.f0200')
            f0_24kHz_file  = os.path.join(cfg.nn_feat_dirs['f024k'], speaker_id, file_id + '.f024k')

            self.DFC.f0_reaper_2_f0_200Hz(f0_reaper_file, f0_200Hz_file)
            self.DFC.f0_200Hz_2_f0_1D(f0_200Hz_file, f0_24kHz_file)

    def reduce_silence(self, feat_name, last_speaker_id=0):
        self.logger.info('reduce_silence %s' % feat_name)
        cfg = self.cfg
        file_id_list = self.file_id_list

        from frontend_mw545.data_silence_reducer import Data_Silence_Reducer
        self.DSR = Data_Silence_Reducer(self.cfg)

        in_file_dir = cfg.nn_feat_dirs[feat_name]
        out_file_dir = cfg.nn_feat_resil_dirs[feat_name]
        file_ext = '.' + feat_name

        self.prepare_speaker_dir(out_file_dir)

        for file_id in self.file_id_list:
            speaker_id = file_id.split('_')[0]
            if int(speaker_id[1:]) > (last_speaker_id-1):
                alignment_file_name = os.path.join(cfg.lab_dir, speaker_id, file_id + '.lab')
                in_file_name  = os.path.join(in_file_dir,  speaker_id, file_id + file_ext)
                out_file_name = os.path.join(out_file_dir, speaker_id, file_id + file_ext)

                self.logger.info('Saving to file %s' % out_file_name)
                self.DSR.reduce_silence_file(alignment_file_name, in_file_name, out_file_name, feat_name=feat_name)

    def cmp_norm_files(self):
        self.logger.info('MVN normalise cmp')
        cfg = self.cfg
        file_id_list = self.file_id_list

        from frontend_mw545.data_norm import Data_Mean_Var_Normaliser
        self.DMVN = Data_Mean_Var_Normaliser(self.cfg)
        # 1. compute mean and std values
        self.DMVN.compute_mean()
        self.DMVN.compute_std()
        # 2. save mean and std values
        self.DMVN.save_mean_std_values()
        # 3. normalise all files
        feat_name = 'cmp'
        in_file_dir = cfg.nn_feat_resil_dirs[feat_name]
        out_file_dir = cfg.nn_feat_resil_norm_dirs[feat_name]
        file_ext = '.' + feat_name
        self.prepare_speaker_dir(out_file_dir)

        for file_id in self.file_id_list:
            speaker_id = file_id.split('_')[0]
            in_file_name  = os.path.join(in_file_dir,  speaker_id, file_id + file_ext)
            out_file_name = os.path.join(out_file_dir, speaker_id, file_id + file_ext)

            self.logger.info('Saving to file %s' % out_file_name)
            self.DMVN.norm_file(in_file_name, out_file_name)

    def wav_norm_files(self, last_speaker_id=0):
        self.logger.info('MM normalise wav')
        cfg = self.cfg
        file_id_list = self.file_id_list

        from frontend_mw545.data_norm import Data_Wav_Min_Max_Normaliser
        self.DWMMN = Data_Wav_Min_Max_Normaliser(self.cfg)

        feat_name = 'wav'
        in_file_dir = cfg.nn_feat_resil_dirs[feat_name]
        out_file_dir = cfg.nn_feat_resil_norm_dirs[feat_name]
        file_ext = '.' + feat_name
        self.prepare_speaker_dir(out_file_dir)

        for file_id in self.file_id_list:
            speaker_id = file_id.split('_')[0]
            if int(speaker_id[1:]) > (last_speaker_id-1):
                in_file_name  = os.path.join(in_file_dir,  speaker_id, file_id + file_ext)
                out_file_name = os.path.join(out_file_dir, speaker_id, file_id + file_ext)

                self.logger.info('Saving to file %s' % out_file_name)
                self.DWMMN.norm_file(in_file_name, out_file_name)

def make_pml_single(inputs):
    wav_file, vocoder_file_dict = inputs
    cfg = configuration()
    DFC = Data_File_Converter(cfg)
    DFC.wav_2_acoustic(wav_file, vocoder_file_dict, acoustic_in_dimension_dict=cfg.acoustic_in_dimension_dict)

class configuration(object):
    def __init__(self, work_dir=None):
        self.init_all(work_dir)

        self.Processes = {}
        # All kinds of tests
        self.Processes['TemporaryTest'] = False
        self.Processes['FrontendTest']  = False
        self.Processes['TorchTest']  = False

        # New Data Preparation Functions
        self.Processes['DataConvert']       = False
        self.Processes['DataSilenceReduce'] = False
        self.Processes['DataNorm']          = False


        self.Processes['TrainCMPDVY'] = False
        self.Processes['TestCMPDVY']  = False

        # 200ms window is sliced into smaller windows of 40ms
        # lf0, tau, vuv are concatenated to the wav vector
        self.Processes['TrainWavSubwinDVY'] = False
        self.Processes['TestWavSubwinDVY']  = False

        # Experiments with SincNet
        self.Processes['TrainWavSincNet'] = False
        self.Processes['TestWavSincNet']  = False

        # Experiments where F0 and phase shift info are predicted
        # 200ms window is sliced into smaller frames of 40ms
        self.Processes['TrainWavSineV0'] = False
        self.Processes['TestWavSineV0']  = False

        self.Processes['TrainWavSineV1'] = True
        self.Processes['TestWavSineV1']  = False

        self.Processes['TrainWavSineV2'] = False
        self.Processes['TestWavSineV2']  = False

    def init_all(self, work_dir):
        if work_dir is None:
            self.work_dir = "/home/dawna/tts/mw545/TorchDV/debug_nausicaa"
        else:
            self.work_dir = work_dir # Comes from bash command argument, ${PWD}

        # self.python_script_name = os.path.join(self.work_dir, 'run_nn_iv_batch_T4_DV.py')
        self.python_script_name = os.path.realpath(__file__)
        prepare_script_file_path(self.work_dir, self.python_script_name)
        
        # self.data_dir = os.path.join(self.work_dir, 'data')
        self.data_dir = '/data/vectra2/tts/mw545/Data/exp_dirs/data_voicebank_24kHz'
        self.file_id_list_dir = os.path.join(self.data_dir, 'file_id_lists')
        
        self.question_file_name = os.path.join(self.data_dir, 'questions.hed')
        # TODO: hard code here; change after vectra2 is fixed
        # self.file_id_list_file  = os.path.join(self.data_dir, 'file_id_list.scp')
        self.file_id_list_file = {}
        self.file_id_list_file['all']  = os.path.join(self.file_id_list_dir, 'file_id_list.scp')    # Complete file id list
        self.file_id_list_file['used'] = os.path.join(self.file_id_list_dir, 'file_id_list_used_cfg.scp') # Only files used in any experiments
        self.file_id_list_file['excluded'] = os.path.join(self.file_id_list_dir, 'file_id_list_not_used_cfg.scp') # Only files never used in any experiments; all=used+excluded
        self.file_id_list_file['compute_norm_info'] = os.path.join(self.file_id_list_dir, 'file_id_list_used_cfg_compute_norm_info.scp') # Files used to compute mean/std, min/max, for normalisation
        self.file_id_list_file['dv_test']  = os.path.join(self.file_id_list_dir, 'file_id_list_dv_test.scp') # DV testing, train speaker, 41-80
        self.file_id_list_file['dv_enough']  = os.path.join(self.file_id_list_dir, 'file_id_list_used_cfg_dv_enough.scp') # Used cfg, and long enough for dv extraction, >= 1s, 200 frames
        self.file_id_list_file['dv_pos_test']  = os.path.join(self.file_id_list_dir, 'file_id_list_used_cfg_dv_pos_test.scp') # DV positional testing, train speaker, 41-80, draw 5 from each speaker

        # Raw data directories
        self.lab_dir = os.path.join(self.data_dir, 'label_state_align')
        self.wav_dir = os.path.join(self.data_dir, 'wav24')
        self.reaper_pitch_dir = os.path.join(self.data_dir, 'reaper_24kHz/pitch') # .pitch
        self.reaper_f0_dir = os.path.join(self.data_dir, 'reaper_24kHz/f0') # .f0

        # Vocoder Features
        self.acoustic_feature_type = 'PML'
        self.output_contain_delta  = False
        self.acoustic_features  = ['mgc', 'lf0', 'bap']
        self.acoustic_in_dimension_dict  = {'mgc':60,  'lf0':1, 'bap':25}
        self.compute_acoustic_out_dimension_dict() # Generate self.acoustic_out_dimension_dict

        # Vocoder directories
        self.acoustic_data_dir = os.path.join(self.data_dir, self.acoustic_feature_type)
        self.acoustic_dir_dict = {}
        for feat_name in self.acoustic_features:
            self.acoustic_dir_dict[feat_name] = os.path.join(self.acoustic_data_dir, feat_name)
        self.acoustic_file_ext_dict  = {'mgc':'.mcep',  'lf0':'.lf0', 'bap':'.bndnm'}
        
        # Parameters
        self.wav_sr  = 24000
        self.synthesis_wav_sr = 24000
        self.frame_sr = 200
        # self.delta_win = [-0.5, 0.0, 0.5]
        # self.acc_win   = [1.0, -2.0, 1.0]
        self.dv_dim = 512

        # Silence to keep (at 200Hz)
        self.frames_silence_to_keep = 0
        self.sil_pad = 5

        self.nn_features     = ['lab', 'cmp', 'wav', 'pitch', 'f024k', 'lf024k']
        self.nn_feature_dims = {}
        self.nn_feature_dims['lab'] = 601
        self.nn_feature_dims['cmp'] = sum(self.acoustic_out_dimension_dict.values())
        self.nn_feature_dims['wav'] = int(self.wav_sr / self.frame_sr)
        self.nn_feature_dims['pitch']  = self.nn_feature_dims['wav']
        self.nn_feature_dims['f024k']  = self.nn_feature_dims['wav']
        self.nn_feature_dims['lf024k'] = self.nn_feature_dims['wav']

        # Features: First numericals, "cmp" style
        self.nn_feat_dirs             = {}
        # Features: Reduce Silence
        self.nn_feat_resil_dirs       = {}
        # Features: Normalisation
        self.nn_feat_resil_norm_dirs  = {}
        self.nn_feat_resil_norm_files = {}
        # Scratch directories for speed-up
        self.nn_feat_scratch_dir_root = '/scratch/tmp-mw545/voicebank_208_speakers_24kHz'
        self.nn_feat_scratch_dirs     = {}
        for nn_feat in self.nn_features:
            self.nn_feat_dirs[nn_feat]             = os.path.join(self.data_dir, 'nn_' + nn_feat)
            self.nn_feat_resil_dirs[nn_feat]       = self.nn_feat_dirs[nn_feat] + '_resil'
            self.nn_feat_resil_norm_dirs[nn_feat]  = self.nn_feat_resil_dirs[nn_feat] + '_norm_' + str(self.nn_feature_dims[nn_feat])
            self.nn_feat_resil_norm_files[nn_feat] = self.nn_feat_resil_norm_dirs[nn_feat] +'_info.dat'
            if nn_feat in ['pitch', 'f024k']:
                # These 2 are not normalised
                self.nn_feat_scratch_dirs[nn_feat] = os.path.join(self.nn_feat_scratch_dir_root, self.nn_feat_resil_dirs[nn_feat].split('/')[-1])
            elif nn_feat in ['lab', 'cmp', 'wav', 'lf024k']:
                self.nn_feat_scratch_dirs[nn_feat] = os.path.join(self.nn_feat_scratch_dir_root, self.nn_feat_resil_norm_dirs[nn_feat].split('/')[-1])

        # self.nn_feat_scratch_dirs['pitch'] = os.path.join(self.nn_feat_scratch_dir_root, 'pitch')

        self.held_out_file_number    = [1,80]
        self.AM_held_out_file_number = [1,40]

        self.make_speaker_id_list_dict()

        self.log_except_list = ['log_except_list', 'all_speaker_list', 'male_speaker_list', 'train_speaker_list', 'valid_speaker_list', 'test_speaker_list', 'speaker_id_list_dict']

    def compute_acoustic_out_dimension_dict(self):
        if self.acoustic_feature_type == 'PML':
            self.acoustic_out_dimension_dict = {}
            for feat_name in self.acoustic_in_dimension_dict.keys():
                if self.output_contain_delta:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name] * 3
                else:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name]
            self.acoustic_start_index = {
                    'mgc':0, 
                    'lf0':self.acoustic_out_dimension_dict['mgc'], 
                    'bap':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['lf0']}
        elif self.acoustic_feature_type == 'STRAIGHT':
            self.acoustic_out_dimension_dict = {'vuv':1}
            for feat_name in self.acoustic_in_dimension_dict.keys():
                if self.output_contain_delta:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name] * 3
                else:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name]
            self.acoustic_start_index = {
                    'mgc':0, 
                    'vuv':self.acoustic_out_dimension_dict['mgc'], 
                    'lf0':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['vuv'],
                    'bap':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['vuv']+self.acoustic_out_dimension_dict['lf0']}

    def make_speaker_id_list_dict(self):
        speaker_id_list_dict = {}
        speaker_id_list_dict['all'] = ['p001', 'p002', 'p003', 'p004', 'p005', 'p006', 'p007', 'p008', 'p010', 'p011', 'p013', 'p014', 'p015', 'p017', 'p019', 'p021', 'p022', 'p023', 'p024', 'p026', 'p027', 'p028', 'p030', 'p031', 'p032', 'p033', 'p034', 'p035', 'p036', 'p037', 'p038', 'p039', 'p043', 'p044', 'p045', 'p047', 'p048', 'p049', 'p052', 'p053', 'p054', 'p055', 'p056', 'p057', 'p060', 'p061', 'p062', 'p063', 'p065', 'p067', 'p068', 'p069', 'p070', 'p071', 'p073', 'p074', 'p075', 'p076', 'p077', 'p079', 'p081', 'p084', 'p085', 'p087', 'p088', 'p089', 'p090', 'p091', 'p093', 'p094', 'p095', 'p096', 'p097', 'p098', 'p099', 'p100', 'p101', 'p102', 'p103', 'p105', 'p106', 'p107', 'p109', 'p110', 'p112', 'p113', 'p114', 'p116', 'p117', 'p118', 'p120', 'p122', 'p123', 'p124', 'p125', 'p126', 'p128', 'p129', 'p130', 'p131', 'p132', 'p134', 'p135', 'p136', 'p139', 'p140', 'p141', 'p142', 'p146', 'p147', 'p151', 'p152', 'p153', 'p155', 'p156', 'p157', 'p158', 'p160', 'p161', 'p162', 'p163', 'p164', 'p165', 'p166', 'p167', 'p168', 'p170', 'p171', 'p173', 'p174', 'p175', 'p176', 'p177', 'p178', 'p179', 'p180', 'p182', 'p184', 'p187', 'p188', 'p192', 'p194', 'p197', 'p200', 'p201', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p215', 'p216', 'p217', 'p218', 'p219', 'p220', 'p221', 'p223', 'p224', 'p290', 'p293', 'p294', 'p295', 'p298', 'p299', 'p300', 'p302', 'p303', 'p304', 'p306', 'p308', 'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p320', 'p321', 'p322', 'p327', 'p331', 'p333', 'p334', 'p336', 'p337', 'p339', 'p340', 'p341', 'p343', 'p344', 'p347', 'p348', 'p349', 'p350', 'p351', 'p353', 'p354', 'p356', 'p370', 'p375', 'p376', 'p384', 'p386', 'p398']
        # p041 has been removed, voice of a sick person
        # p202 is not in file_id_list yet, and he has same voice as p209, be careful
        speaker_id_list_dict['valid'] = ['p162', 'p002', 'p303', 'p048', 'p109', 'p153', 'p038', 'p166', 'p218', 'p070']    # Last 3 are males
        speaker_id_list_dict['test']  = ['p293', 'p210', 'p026', 'p024', 'p313', 'p223', 'p141', 'p386', 'p178', 'p290'] # Last 3 are males
        speaker_id_list_dict['not_train'] = speaker_id_list_dict['valid']+speaker_id_list_dict['test']
        speaker_id_list_dict['train'] = [spk for spk in speaker_id_list_dict['all'] if (spk not in speaker_id_list_dict['not_train'])]
        speaker_id_list_dict['male']  = ['p001', 'p015', 'p033', 'p065', 'p004', 'p010', 'p094', 'p099', 'p102', 'p039', 'p136', 'p007', 'p151', 'p028', 'p019', 'p070', 'p192', 'p017', 'p101', 'p096', 'p014', 'p006', 'p087', 'p063', 'p079', 'p134', 'p116', 'p088', 'p030', 'p003', 'p157', 'p031', 'p118', 'p076', 'p171', 'p177', 'p180', 'p036', 'p126', 'p179', 'p215', 'p212', 'p219', 'p218', 'p173', 'p194', 'p209', 'p174', 'p166', 'p178', 'p130', 'p344', 'p334', 'p347', 'p302', 'p298', 'p304', 'p311', 'p316', 'p322', 'p224', 'p290', 'p320', 'p356', 'p375', 'p386', 'p376', 'p384', 'p398']

        for k in speaker_id_list_dict:
            speaker_id_list_dict[k].sort()

        self.speaker_id_list_dict = speaker_id_list_dict
        self.num_speaker_dict = {k: len(self.speaker_id_list_dict[k]) for k in ['all', 'train', 'valid', 'test', 'male']}



def main_function(cfg):

    logger = make_logger("Main_config")
    logger.info('PID is %i' % os.getpid())
    log_class_attri(cfg, logger, except_list=cfg.log_except_list)

    logger = make_logger("Main")

    if cfg.Processes['TemporaryTest']:
        from exp_mw545.exp_dv_temp_test import temporary_test
        temporary_test(cfg)

    if cfg.Processes['FrontendTest']:
        from frontend_mw545.frontend_tests import run_Frontend_Test
        run_Frontend_Test(cfg)

    if cfg.Processes['TorchTest']:
        from nn_torch.torch_tests import run_Torch_Test
        run_Torch_Test(cfg)
        

    if cfg.Processes['DataConvert']:
        from frontend_mw545.data_converter import run_Data_File_List_Converter
        run_Data_File_List_Converter(cfg)

    if cfg.Processes['DataSilenceReduce']:
        from frontend_mw545.data_silence_reducer import run_Data_Silence_List_Reducer
        run_Data_Silence_List_Reducer(cfg)

    if cfg.Processes['DataNorm']:
        from frontend_mw545.data_norm import run_Data_List_Normaliser
        run_Data_List_Normaliser(cfg)

        
        

    if cfg.Processes['TrainCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import train_model
        train_model(cfg)

    if cfg.Processes['TestCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSubwinDVY']:
        from exp_mw545.exp_dv_wav_subwin import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSubwinDVY']:
        from exp_mw545.exp_dv_wav_subwin import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSincNet']:
        from exp_mw545.exp_dv_wav_sincnet import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSincNet']:
        from exp_mw545.exp_dv_wav_sincnet import test_model
        test_model(cfg)


    if cfg.Processes['TrainWavSineV0']:
        from exp_mw545.exp_dv_wav_sinenet_v0 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV0']:
        from exp_mw545.exp_dv_wav_sinenet_v0 import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV1']:
        from exp_mw545.exp_dv_wav_sinenet_v1 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV1']:
        from exp_mw545.exp_dv_wav_sinenet_v1 import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV2']:
        from exp_mw545.exp_dv_wav_sinenet_v2 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV2']:
        from exp_mw545.exp_dv_wav_sinenet_v2 import test_model
        test_model(cfg)




if __name__ == '__main__': 

    if len(sys.argv) == 2:
        work_dir = sys.argv[1]
    else:
        work_dir = None

    cfg = configuration(work_dir)
    # main_function(cfg)
    p = prep_24kHz_dir(cfg)
    p.run()

