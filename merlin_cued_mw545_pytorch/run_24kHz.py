import os, sys, pickle, time, shutil, logging, importlib
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

    if cfg.Processes['Train']:
        mymodule = importlib.import_module("scripts."+cfg.script_name)
        mymodule.train_model(cfg)

    if cfg.Processes['Test']:
        mymodule = importlib.import_module("scripts."+cfg.script_name)
        mymodule.test_model(cfg)
    

if __name__ == '__main__': 

    work_dir = sys.argv[1]
    config_file = sys.argv[2]

    cfg = configuration(work_dir, config_file)
    main_function(cfg)
