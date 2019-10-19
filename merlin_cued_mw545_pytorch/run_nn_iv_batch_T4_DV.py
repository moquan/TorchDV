import os, sys, pickle, time, shutil, logging
import math, numpy, scipy, scipy.io.wavfile#, sigproc, sigproc.pystraight

from modules import make_logger, read_file_list, prepare_file_path, prepare_file_path_list, make_held_out_file_number, copy_to_scratch
from modules import keep_by_speaker, remove_by_speaker, keep_by_file_number, remove_by_file_number
from modules_2 import log_class_attri, resil_nn_file_list, norm_nn_file_list

class configuration(object):
    def __init__(self, work_dir=None):
        if work_dir is None:
            self.work_dir = "/home/dawna/tts/mw545/TorchDV/debug"
        else:
            self.work_dir = work_dir # Comes from bash command argument, ${PWD}
        self.Processes = {}
        self.Processes['copy_to_scratch'] = False
        self.Processes['MakeCmp']  = False
        self.Processes['MakeWav']  = False
        self.Processes['ResilLab'] = False
        self.Processes['ResilCmp'] = False
        self.Processes['ResilWav'] = False
        self.Processes['NormLab']  = False
        self.Processes['NormCmp']  = False
        self.Processes['NormWav']  = False
        # self.Processes['MuLawWav'] = False
        self.Processes['ResilPitch']   = False

        self.Processes['TrainCMPTorch'] = True
        self.Processes['TestCMPTorch']  = True

        self.Processes['TrainCMPDVY'] = False
        self.Processes['TestCMPDVY']  = False
        self.Processes['GenCMPDVY']   = False


        self.Processes['TrainWavDVY'] = False
        self.Processes['TestWavDVY']  = False
        self.Processes['GenWavDVY']   = False


        self.Processes['TrainWavCA'] = False
        self.Processes['TestWavCA']  = False
        self.Processes['GenWavCA']   = False
        self.Processes['GenWavAttenCA'] = False

        # Experiments where REAPER F0 and phase shift info are known
        self.Processes['TrainWavSine'] = False
        self.Processes['TestWavSine']  = False
        self.Processes['GenWavSine']   = False


        self.Processes['TrainAM'] = False
        self.Processes['GenAM']   = False
        self.Processes['CMP2Wav'] = False
        self.Processes['CalMCD']  = False


        self.Processes['wavcmpCheck'] = False
        self.Processes['cmpwavCheck'] = False
        self.Processes['remakePML']   = False

        
        prepare_file_path(self.work_dir)

        self.dv_dim = 8

        self.python_script_name = os.path.join(self.work_dir, 'run_nn_iv_batch_T4_DV.py')
        
        # self.data_dir = os.path.join(self.work_dir, 'data')
        self.data_dir = '/data/vectra2/tts/mw545/Data/data_voicebank'
        
        self.question_file_name = os.path.join(self.data_dir, 'questions.hed')
        # TODO: hard code here; change after vectra2 is fixed
        self.file_id_list_file  = os.path.join(self.data_dir, 'file_id_list.scp')
        self.file_id_list_file  = os.path.join('/home/dawna/tts/mw545/TorchDV', 'file_id_list.scp')
        self.frames_silence_to_keep = 50
        self.sil_pad = 5
        self.delta_win = [-0.5, 0.0, 0.5]
        self.acc_win   = [1.0, -2.0, 1.0]

        # Features: Component directories
        self.lab_dir = os.path.join(self.data_dir, 'label_state_align')
        self.acoustic_feature_type = 'PML'
        self.output_contain_delta  = False
        self.acoustic_features  = ['mgc', 'lf0', 'bap']
        self.acoustic_in_dimension_dict  = {'mgc':60,  'lf0':1, 'bap':25}
        if self.acoustic_feature_type == 'PML':
            self.acoustic_out_dimension_dict = {}
            for feat_name in self.acoustic_in_dimension_dict.keys():
                if self.output_contain_delta:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name] * 3
                else:
                    self.acoustic_out_dimension_dict[feat_name] = self.acoustic_in_dimension_dict[feat_name]
            self.acoustic_start_index        = {
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
            self.acoustic_start_index        = {
                        'mgc':0, 
                        'lf0':self.acoustic_out_dimension_dict['mgc'], 
                        'bap':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['lf0']}
            self.acoustic_start_index        = {
                        'mgc':0, 
                        'vuv':self.acoustic_out_dimension_dict['mgc'], 
                        'lf0':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['vuv'],
                        'bap':self.acoustic_out_dimension_dict['mgc']+self.acoustic_out_dimension_dict['vuv']+self.acoustic_out_dimension_dict['lf0']}
        self.acoustic_file_ext_dict  = {'mgc':'.mcep',  'lf0':'.lf0', 'bap':'.bndnm'}

        self.acoustic_data_dir  = os.path.join(self.data_dir, self.acoustic_feature_type)
        self.acoustic_dir_dict  = {}
        self.var_file_dict      = {}
        for feat_name in self.acoustic_features:
            self.acoustic_dir_dict[feat_name] = os.path.join(self.acoustic_data_dir, feat_name)
            self.var_file_dict[feat_name] = os.path.join(self.acoustic_data_dir, 'var', feat_name+'_'+str(self.acoustic_out_dimension_dict[feat_name]))
            
        self.wav_dir = os.path.join(self.data_dir, 'wav_16kHz')
        self.wav_sr  = 16000
        self.synthesis_wav_sr = 16000
        self.frame_sr = 200

        self.nn_features     = ['lab', 'cmp', 'wav']
        self.nn_feature_dims = {}
        self.nn_feature_dims['lab'] = 601
        self.nn_feature_dims['cmp'] = sum(self.acoustic_out_dimension_dict.values())
        self.nn_feature_dims['wav'] = self.wav_sr / self.frame_sr

        # Features: First numericals
        self.nn_feat_dirs             = {}
        # Features: Reduce Silence
        self.nn_feat_resil_dirs       = {}
        # Features: Normalisation
        self.nn_feat_resil_norm_dirs  = {}
        self.nn_feat_resil_norm_files = {}
        # Scratch directories for speed-up
        self.nn_feat_scratch_dir_root = '/scratch/tmp-mw545/voicebank_208_speakers'        
        self.nn_feat_scratch_dirs     = {}
        for nn_feat in self.nn_features:
            self.nn_feat_dirs[nn_feat]             = os.path.join(self.data_dir, 'nn_' + nn_feat)
            self.nn_feat_resil_dirs[nn_feat]       = self.nn_feat_dirs[nn_feat] + '_resil'
            self.nn_feat_resil_norm_dirs[nn_feat]  = self.nn_feat_resil_dirs[nn_feat] + '_norm_' + str(self.nn_feature_dims[nn_feat])
            self.nn_feat_resil_norm_files[nn_feat] = self.nn_feat_resil_norm_dirs[nn_feat] +'_info.dat'
            self.nn_feat_scratch_dirs[nn_feat]     = os.path.join(self.nn_feat_scratch_dir_root, self.nn_feat_resil_norm_dirs[nn_feat].split('/')[-1])
        self.nn_feat_scratch_dirs['pitch'] = os.path.join(self.nn_feat_scratch_dir_root, 'pitch')

        self.held_out_file_number = make_held_out_file_number(80)
        self.AM_held_out_file_number = make_held_out_file_number(40)

        self.speaker_id_list_dict, self.num_speaker_dict = self.make_speaker_id_list_dict()

        self.log_except_list = ['all_speaker_list', 'male_speaker_list', 'train_speaker_list', 'valid_speaker_list', 'test_speaker_list', 'speaker_id_list_dict', 'held_out_file_number', 'AM_held_out_file_number']


    def make_speaker_id_list_dict(self):
        speaker_id_list_dict = {}
        speaker_id_list_dict['all'] = ['p100', 'p101', 'p102', 'p103', 'p105', 'p106', 'p107', 'p109', 'p10', 'p110', 'p112', 'p113', 'p114', 'p116', 'p117', 'p118', 'p11', 'p120', 'p122', 'p123', 'p124', 'p125', 'p126', 'p128', 'p129', 'p130', 'p131', 'p132', 'p134', 'p135', 'p136', 'p139', 'p13', 'p140', 'p141', 'p142', 'p146', 'p147', 'p14', 'p151', 'p152', 'p153', 'p155', 'p156', 'p157', 'p158', 'p15', 'p160', 'p161', 'p162', 'p163', 'p164', 'p165', 'p166', 'p167', 'p168', 'p170', 'p171', 'p173', 'p174', 'p175', 'p176', 'p177', 'p178', 'p179', 'p17', 'p180', 'p182', 'p184', 'p187', 'p188', 'p192', 'p194', 'p197', 'p19', 'p1', 'p200', 'p201', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p215', 'p216', 'p217', 'p218', 'p219', 'p21', 'p220', 'p221', 'p223', 'p224', 'p22', 'p23', 'p24', 'p26', 'p27', 'p28', 'p290', 'p293', 'p294', 'p295', 'p298', 'p299', 'p2', 'p300', 'p302', 'p303', 'p304', 'p306', 'p308', 'p30', 'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p31', 'p320', 'p321', 'p322', 'p327', 'p32', 'p331', 'p333', 'p334', 'p336', 'p337', 'p339', 'p33', 'p340', 'p341', 'p343', 'p344', 'p347', 'p348', 'p349', 'p34', 'p350', 'p351', 'p353', 'p354', 'p356', 'p35', 'p36', 'p370', 'p375', 'p376', 'p37', 'p384', 'p386', 'p38', 'p398', 'p39', 'p3', 'p41', 'p43', 'p44', 'p45', 'p47', 'p48', 'p49', 'p4', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p5', 'p60', 'p61', 'p62', 'p63', 'p65', 'p67', 'p68', 'p69', 'p6', 'p70', 'p71', 'p73', 'p74', 'p75', 'p76', 'p77', 'p79', 'p7', 'p81', 'p84', 'p85', 'p87', 'p88', 'p89', 'p8', 'p90', 'p91', 'p93', 'p94', 'p95', 'p96', 'p97', 'p98', 'p99']        
        speaker_id_list_dict['valid'] = ['p162', 'p2', 'p303', 'p48', 'p109', 'p153', 'p38', 'p166', 'p218', 'p70']    # Last 3 are males
        speaker_id_list_dict['test']  = ['p293', 'p210', 'p26', 'p24', 'p313', 'p223', 'p141', 'p386', 'p178', 'p290'] # Last 3 are males
        speaker_id_list_dict['train'] = [spk for spk in speaker_id_list_dict['all'] if (spk not in (speaker_id_list_dict['valid']+speaker_id_list_dict['test']))]
        speaker_id_list_dict['male']  = ['p1', 'p15', 'p33', 'p65', 'p4', 'p10', 'p94', 'p99', 'p102', 'p39', 'p136', 'p7', 'p151', 'p28', 'p19', 'p70', 'p192', 'p17', 'p101', 'p96', 'p14', 'p6', 'p87', 'p63', 'p79', 'p134', 'p116', 'p88', 'p30', 'p3', 'p157', 'p31', 'p118', 'p76', 'p171', 'p177', 'p180', 'p36', 'p126', 'p179', 'p215', 'p212', 'p219', 'p218', 'p173', 'p194', 'p209', 'p174', 'p166', 'p178', 'p130', 'p344', 'p334', 'p347', 'p302', 'p298', 'p304', 'p311', 'p316', 'p322', 'p224', 'p290', 'p320', 'p356', 'p375', 'p386', 'p376', 'p384', 'p398']

        num_speaker_dict = {}
        for k in ['all', 'train', 'valid', 'test', 'male']:
            num_speaker_dict[k] = len(speaker_id_list_dict[k])

        return speaker_id_list_dict, num_speaker_dict


    def need_to_load_file_id_list(self):
        need_list = ['copy_to_scratch', 'MakeCmp', 'MakeWav', 'ResilLab', 'ResilCmp', 'ResilWav', 'ResilPitch', 'NormLab', 'NormCmp', 'NormWav', 'remakePML']
        for process_name in need_list:
            if self.Processes[process_name]:
                return True
        return False

def main_function(cfg):

    logger = make_logger("Main_config")
    log_class_attri(cfg, logger, except_list=cfg.log_except_list)

    logger = make_logger("Main")

    if cfg.need_to_load_file_id_list():
        file_id_list = read_file_list(cfg.file_id_list_file)
        nn_file_list = {}
        nn_resil_file_list = {}
        nn_resil_norm_file_list = {}

    if cfg.Processes['copy_to_scratch']:
        logger.info('copy_to_scratch')
        copy_to_scratch(cfg, file_id_list)

    if cfg.Processes['MakeCmp']:
        logger.info('MakeCmp')
        from modules import acoustic_2_cmp_list
        acoustic_file_list_dict = {}
        for feat in cfg.acoustic_features:
            acoustic_file_list_dict[feat] = prepare_file_path_list(file_id_list, cfg.acoustic_dir_dict[feat], cfg.acoustic_file_ext_dict[feat])
        nn_file_list['cmp'] = prepare_file_path_list(file_id_list, cfg.nn_feat_dirs['cmp'], '.cmp')
        acoustic_2_cmp_list(cfg, acoustic_file_list_dict, nn_file_list['cmp'])

    if cfg.Processes['MakeWav']:
        logger.info('MakeWav')
        from modules import wav_2_wav_cmp_list
        wav_file_list = prepare_file_path_list(file_id_list, cfg.wav_dir, '.wav')
        nn_file_list['wav'] = prepare_file_path_list(file_id_list, cfg.nn_feat_dirs['wav'], '.wav')
        wav_2_wav_cmp_list(wav_file_list, nn_file_list['wav'])

    if cfg.Processes['ResilLab']:
        logger.info('ResilLab')
        resil_nn_file_list('lab', cfg, file_id_list, nn_file_list, nn_resil_file_list)

    if cfg.Processes['ResilCmp']:
        logger.info('ResilCmp')
        resil_nn_file_list('cmp', cfg, file_id_list, nn_file_list, nn_resil_file_list)

    if cfg.Processes['ResilWav']:
        logger.info('ResilWav')
        resil_nn_file_list('wav', cfg, file_id_list, nn_file_list, nn_resil_file_list)

    if cfg.Processes['ResilPitch']:
        logger.info('ResilPitch')
        from modules import reduce_silence_reaper_output_list
        reaper_output_dir = '/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch'
        label_align_dir   = '/data/vectra2/tts/mw545/Data/data_voicebank/label_state_align'
        out_dir           = '/home/dawna/tts/mw545/Data/Data_Voicebank_48kHz_Pitch_Resil'
        reduce_silence_reaper_output_list(cfg, file_id_list, reaper_output_dir, label_align_dir, out_dir, reaper_output_ext='.used.pm', label_align_ext='.lab', out_ext='.pm')

    if cfg.Processes['NormLab']:
        logger.info('NormLab')
        norm_nn_file_list('lab', cfg, file_id_list, nn_resil_file_list, nn_resil_norm_file_list, compute_normaliser=True, norm_type='MinMax')    
        
    if cfg.Processes['NormCmp']:
        logger.info('NormCmp')
        norm_nn_file_list('cmp', cfg, file_id_list, nn_resil_file_list, nn_resil_norm_file_list, compute_normaliser=True, norm_type='MeanVar')

    if cfg.Processes['NormWav']:
        logger.info('NormWav')
        norm_nn_file_list('wav', cfg, file_id_list, nn_resil_file_list, nn_resil_norm_file_list, compute_normaliser=True, norm_type='MinMax')

    # if cfg.Processes['MuLawWav']:
    #     logger.info('MuLawWav')
    #     from modules import perform_mu_law_list
    #     feat_name = 'wav'
    #     nn_resil_norm_file_list[feat_name] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.'+feat_name)
    #     nn_resil_norm_file_list[feat_name+'_mu'] = prepare_file_path_list(file_id_list, cfg.nn_feat_resil_norm_dirs[feat_name], '.mu.'+feat_name)
    #     perform_mu_law_list(nn_resil_norm_file_list[feat_name], nn_resil_norm_file_list[feat_name+'_mu'], mu_value=255.)

    if cfg.Processes['TrainCMPTorch']:
        from exp_mw545.exp_dv_cmp_pytorch import train_dv_y_cmp_model
        train_dv_y_cmp_model(cfg)

    if cfg.Processes['TestCMPTorch']:
        from exp_mw545.exp_dv_cmp_pytorch import test_dv_y_cmp_model
        test_dv_y_cmp_model(cfg)



    if cfg.Processes['TrainCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import train_dv_y_cmp_model
        train_dv_y_cmp_model(cfg)

    if cfg.Processes['TestCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import test_dv_y_cmp_model
        test_dv_y_cmp_model(cfg)

    if cfg.Processes['GenCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import gen_dv_y_cmp_model
        gen_dv_y_cmp_model(cfg)



        

    if cfg.Processes['TrainWavDVY']:
        from exp_mw545.exp_dv_wav_baseline import train_dv_y_model
        train_dv_y_model(cfg)

    if cfg.Processes['TestWavDVY']:
        from exp_mw545.exp_dv_wav_baseline import test_dv_y_model
        test_dv_y_model(cfg)

    if cfg.Processes['GenWavDVY']:
        from exp_mw545.exp_dv_wav_baseline import gen_dv_y_model
        gen_dv_y_model(cfg)

    if cfg.Processes['TrainWavCA']:
        from exp_mw545.exp_dv_wav_cnn_atten import train_dv_y_model
        train_dv_y_model(cfg)

    if cfg.Processes['TestWavCA']:
        from exp_mw545.exp_dv_wav_cnn_atten import test_dv_y_model
        test_dv_y_model(cfg)

    if cfg.Processes['GenWavCA']:
        from exp_mw545.exp_dv_wav_cnn_atten import gen_dv_y_model
        gen_dv_y_model(cfg)

    if cfg.Processes['GenWavAttenCA']:
        from exp_mw545.exp_dv_wav_cnn_atten import gen_atten_traj_dv_y_model
        gen_atten_traj_dv_y_model(cfg)


        

    if cfg.Processes['TrainWavSine']:
        from exp_mw545.exp_dv_wav_sine_atten import train_dv_y_model
        train_dv_y_model(cfg)

    if cfg.Processes['TestWavSine']:
        from exp_mw545.exp_dv_wav_sine_atten import test_dv_y_model
        test_dv_y_model(cfg)

    if cfg.Processes['GenWavSine']:
        from exp_mw545.exp_dv_wav_sine_atten import gen_dv_y_model
        gen_dv_y_model(cfg)






    if cfg.Processes['TrainAM']:
        from exp_mw545.exp_am_baseline import train_am_model
        train_am_model(cfg)

    if cfg.Processes['GenAM']:
        from exp_mw545.exp_am_baseline import gen_am_model
        gen_am_model(cfg)

    if cfg.Processes['CMP2Wav']:
        from exp_mw545.exp_am_baseline import cmp_2_wav
        cmp_2_wav(cfg)

    if cfg.Processes['CalMCD']:
        from exp_mw545.exp_am_baseline import cal_mcd
        cal_mcd(cfg)


    file_id = 'p290_028'
    if cfg.Processes['wavcmpCheck']:
        from modules import wav_2_norm_cmp
        wav_file = "/home/dawna/tts/mw545/TorchDV/debug/data/wav_16kHz/%s.wav" % file_id
        target_dir = "/home/dawna/tts/mw545/TorchDV/debug/wavcmpCheck/make"
        lab_file = "/home/dawna/tts/mw545/TorchDV/debug/data/label_state_align/%s.lab" % file_id
        cmp_norm_file = "/home/dawna/tts/mw545/TorchDV/debug/data/nn_cmp_resil_norm_86_info.dat"
        wav_2_norm_cmp(cfg, wav_file, target_dir, lab_file, cmp_norm_file)

    if cfg.Processes['cmpwavCheck']:
        from modules import norm_cmp_2_wav
        cmp_resil_norm_file = "/home/dawna/tts/mw545/TorchDV/debug/wavcmpCheck/make/%s.cmp.resil.norm" % file_id
        target_dir = "/home/dawna/tts/mw545/TorchDV/debug/wavcmpCheck/gen"
        cmp_norm_file = "/home/dawna/tts/mw545/TorchDV/debug/data/nn_cmp_resil_norm_86_info.dat"
        norm_cmp_2_wav(cfg, cmp_resil_norm_file, target_dir, cmp_norm_file)


    if cfg.Processes['remakePML']:
        from modules import wav_2_acoustic
        # file_id_list_new = keep_by_speaker(file_id_list, cfg.valid_speaker_list + cfg.test_speaker_list)
        for file_id in file_id_list:
            wav_file = "/home/dawna/tts/mw545/TorchDV/debug/data/wav_16kHz/%s.wav" % file_id
            out_file_dict = {}
            for feat_name in cfg.acoustic_features:
                out_file_dict[feat_name] = os.path.join(cfg.acoustic_dir_dict[feat_name], file_id + cfg.acoustic_file_ext_dict[feat_name])
            wav_2_acoustic(wav_file, out_file_dict, cfg.acoustic_in_dimension_dict, verbose_level=0)
        
        


if __name__ == '__main__': 
       

    if len(sys.argv) == 2:
        work_dir = sys.argv[1]
    else:
        work_dir = None

    cfg = configuration(work_dir)

    logger = make_logger("PID")
    logger.info('PID is %i' % os.getpid())
    

    main_function(cfg)

