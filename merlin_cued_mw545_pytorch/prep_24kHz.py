import os, sys, pickle, time, shutil, logging
import math, numpy, scipy, scipy.io.wavfile#, sigproc, sigproc.pystraight

from frontend_mw545.modules import make_logger, read_file_list, prepare_script_file_path, log_class_attri
from config_24kHz import configuration

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

def make_pml_single(inputs):
    wav_file, vocoder_file_dict = inputs
    cfg = configuration()
    DFC = Data_File_Converter(cfg)
    DFC.wav_2_acoustic(wav_file, vocoder_file_dict, acoustic_in_dimension_dict=cfg.acoustic_in_dimension_dict)

if __name__ == '__main__': 

    if len(sys.argv) == 2:
        work_dir = sys.argv[1]
    else:
        work_dir = None

    cfg = configuration(work_dir)
    p = prep_24kHz_dir(cfg)
    p.run()

