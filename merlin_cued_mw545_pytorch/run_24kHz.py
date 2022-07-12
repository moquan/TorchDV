import os, sys, pickle, time, shutil, logging
import math, numpy, scipy, scipy.io.wavfile#, sigproc, sigproc.pystraight

from frontend_mw545.modules import make_logger, prepare_script_file_path, log_class_attri
from config_24kHz import configuration

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
        
    #########################
    # Acoustic-based Models #
    # Flat attention        #
    #########################

    if cfg.Processes['TrainCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import train_model
        train_model(cfg)

    if cfg.Processes['TestCMPDVY']:
        from exp_mw545.exp_dv_cmp_baseline import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSincNet']:
        from exp_mw545.exp_dv_wav_sincnet import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSincNet']:
        from exp_mw545.exp_dv_wav_sincnet import test_model
        test_model(cfg)

    if cfg.Processes['TrainMFCCXVec']:
        from exp_mw545.exp_dv_mfcc_xvector import train_model
        train_model(cfg)

    if cfg.Processes['TestMFCCXVec']:
        from exp_mw545.exp_dv_mfcc_xvector import test_model
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

    ##############################
    # Lab Attention-based Models #
    ##############################

    if cfg.Processes['TrainCMPLabAtten']:
        from exp_mw545.exp_dv_cmp_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestCMPLabAtten']:
        from exp_mw545.exp_dv_cmp_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSincNetLabAtten']:
        from exp_mw545.exp_dv_wav_sincnet_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSincNetLabAtten']:
        from exp_mw545.exp_dv_wav_sincnet_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV0LabAtten']:
        from exp_mw545.exp_dv_wav_sinenet_v0_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV0LabAtten']:
        from exp_mw545.exp_dv_wav_sinenet_v0_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV2LabAtten']:
        from exp_mw545.exp_dv_wav_sinenet_v2_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV2LabAtten']:
        from exp_mw545.exp_dv_wav_sinenet_v2_lab_attention import test_model
        test_model(cfg)




    

if __name__ == '__main__': 

    if len(sys.argv) == 2:
        work_dir = sys.argv[1]
    else:
        work_dir = None

    cfg = configuration(work_dir)
    main_function(cfg)
