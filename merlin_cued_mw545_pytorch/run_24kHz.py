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
        # TODO: add temporary tests here
        pass

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
        from scripts.exp_dv_cmp import train_model
        train_model(cfg)

    if cfg.Processes['TestCMPDVY']:
        from scripts.exp_dv_cmp import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSincNet']:
        from scripts.exp_dv_wav_sincnet import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSincNet']:
        from scripts.exp_dv_wav_sincnet import test_model
        test_model(cfg)

    if cfg.Processes['TrainMFCCXVec']:
        from scripts.exp_dv_mfcc_xvector import train_model
        train_model(cfg)

    if cfg.Processes['TestMFCCXVec']:
        from scripts.exp_dv_mfcc_xvector import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV0']:
        from scripts.exp_dv_wav_sinenet_v0 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV0']:
        from scripts.exp_dv_wav_sinenet_v0 import test_model
        test_model(cfg)
    
    if cfg.Processes['TrainWavSineV1']:
        from scripts.exp_dv_wav_sinenet_v1 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV1']:
        from scripts.exp_dv_wav_sinenet_v1 import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV2']:
        from scripts.exp_dv_wav_sinenet_v2 import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV2']:
        from scripts.exp_dv_wav_sinenet_v2 import test_model
        test_model(cfg)

    ##############################
    # Lab Attention-based Models #
    ##############################

    if cfg.Processes['TrainCMPLabAtten']:
        from scripts.exp_dv_cmp_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestCMPLabAtten']:
        from scripts.exp_dv_cmp_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSincNetLabAtten']:
        from scripts.exp_dv_wav_sincnet_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSincNetLabAtten']:
        from scripts.exp_dv_wav_sincnet_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV0LabAtten']:
        from scripts.exp_dv_wav_sinenet_v0_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV0LabAtten']:
        from scripts.exp_dv_wav_sinenet_v0_lab_attention import test_model
        test_model(cfg)

    if cfg.Processes['TrainWavSineV2LabAtten']:
        from scripts.exp_dv_wav_sinenet_v2_lab_attention import train_model
        train_model(cfg)

    if cfg.Processes['TestWavSineV2LabAtten']:
        from scripts.exp_dv_wav_sinenet_v2_lab_attention import test_model
        test_model(cfg)




    

if __name__ == '__main__': 

    work_dir = sys.argv[1]
    config_file = sys.argv[2]

    cfg = configuration(work_dir, config_file)
    main_function(cfg)
