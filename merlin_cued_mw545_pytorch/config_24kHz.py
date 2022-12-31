import os, json
from frontend_mw545.modules import prepare_script_file_path

class configuration(object):
    def __init__(self, work_dir=None, config_file="", cache_files=True):
        self.load_json_config(config_file)
        self.init_all(work_dir, cache_files)

    def load_json_config(self, config_file):
        json_conf = json.load(open(config_file, ))
        
        json_base_conf = json.load(open(json_conf["baseConfig"], ))
        self.data_dir = json_base_conf["dataDir"]
        self.result_dir = json_base_conf["resultDir"]
        
        self.run_mode = json_conf["mode"]
        self.init_processes(json_conf["processes"].split("|"))
        self.test_list = json_conf["tests"].split("|")
        self.script_name = json_conf["script"]

    def init_processes(self, processes_true):
        self.Processes = {}
        process_list = []
        # All kinds of tests
        process_list.extend(['TemporaryTest', 'FrontendTest', 'TorchTest'])

        # New Data Preparation Functions
        process_list.extend(['DataConvert', 'DataSilenceReduce', 'DataNorm'])

        # train and test
        process_list.extend(['Train', 'Test'])
        '''
        # Vocoder-based d-vector
        process_list.extend(['TrainCMP', 'TestCMP'])
        # Wav-based SincNet
        process_list.extend(['TrainWavSincNet', 'TestWavSincNet'])
        # MFCC-based x-vector: TODO not implemented yet
        process_list.extend(['TrainMFCCXVec', 'TestMFCCXVec'])
        # Experiments where F0 and phase shift info are predicted
        # 200ms window is sliced into smaller frames of 40ms
        process_list.extend(['TrainWavSineV0', 'TestWavSineV0'])
        process_list.extend(['TrainWavSineV1', 'TestWavSineV1'])
        process_list.extend(['TrainWavSineV2', 'TestWavSineV2'])
        
        # Lab-based Attention
        process_list.extend(['TrainCMPLabAtten', 'TestCMPLabAtten'])
        process_list.extend(['TrainWavSincNetLabAtten', 'TestWavSincNetLabAtten'])
        process_list.extend(['TrainWavSineV0LabAtten', 'TestWavSineV0LabAtten'])
        process_list.extend(['TrainWavSineV2LabAtten', 'TestWavSineV2LabAtten'])
        '''

        for p in process_list:
            self.Processes[p] = False

        for p in processes_true:
            self.Processes[p] = True

    def init_all(self, work_dir, cache_files=True):
        if work_dir is None:
            # self.work_dir = "/home/dawna/tts/mw545/TorchDV/debug_nausicaa"
            self.work_dir = "/data/vectra2/tts/mw545/TorchDV"
        else:
            self.work_dir = work_dir # Comes from bash command argument, ${PWD}

        

        self.python_script_name = os.path.realpath(__file__)
        if cache_files:
            prepare_script_file_path(self.work_dir, self.python_script_name)
        
        # self.data_dir = os.path.join(self.work_dir, 'data')
        
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
        self.nn_feat_scratch_dir_root = '/scratch/tmp-mw545/exp_dirs/data_voicebank_24kHz'
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
        self.data_split_file_number = {}
        self.data_split_file_number['train'] = [81, 3000]
        self.data_split_file_number['valid_SR'] = [41, 00]
        self.data_split_file_number['valid_AM']  = [1, 40]

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
