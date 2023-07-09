# sinenet_f_tau_test.py

import os, sys, pickle, time, shutil, logging, copy
import math, numpy, scipy, scipy.stats

from frontend_mw545.modules import make_logger, read_file_list, log_class_attri, Graph_Plotting, List_Random_Loader, File_List_Selecter
from frontend_mw545.data_io import Data_File_IO, Data_List_File_IO, Data_Meta_List_File_IO

from nn_torch.torch_models import Build_DV_Y_model
from frontend_mw545.data_loader import Build_dv_TTS_selecter, Build_dv_y_train_data_loader

from exp_mw545.exp_base import Build_Model_Trainer_Base, DV_Calculator
from exp_mw545.exp_dv_y import Build_DV_Y_Testing_Base

import torch

#########################################
# Test if f and tau in data are optimal #
#########################################

class Build_SineNet_F_Tau_Test(Build_DV_Y_Testing_Base):
    """ 
    use test files only
    compute win_ce, utter_accuracy, win_accuracy 
    """
    def __init__(self, cfg, dv_y_cfg):
        super().__init__(cfg, dv_y_cfg, load_model=True, load_data_loader=True)
        self.logger.info('SineNet_F_Tau_Test')
        # use test files only
        self.speaker_id_list = self.cfg.speaker_id_list_dict['train']

        self.file_id_list = read_file_list(cfg.file_id_list_file['dv_pos_test']) # 186^5 files
        self.cmp_dir = '/data/mifs_scratch/mjfg/mw545/dv_pos_test/cmp_shift_resil_norm'

        self.max_distance   = 50

        self.dv_calculator = DV_Calculator()
        self.graph_plotter = Graph_Plotting()

        self.T_M = 240
        self.M_stride = 240
        self.wav_sr = 24000

    def run(self):
        self.run_3()

    def make_feed_dict_B1(self, feed_dict, i=0):
        # Make a feed_dict, B=1
        new_batch_size = 1
        new_feed_dict = {}

        for k in feed_dict:
            if k == 'h':
                h = feed_dict['h']
                D = h.shape[2]
                h_new = numpy.zeros((1,1,D))
                h_new[0,0] = h[0,i]
                new_feed_dict['h'] = h_new
            elif k in ['output_mask_S_B', 'one_hot_S_B']:
                new_feed_dict[k] = numpy.array([[1.]])
            else:
                new_feed_dict[k] = numpy.array([1.])

        return new_feed_dict, new_batch_size


    def run_3(self):
        # Plot frame-level waveform with filters
        # file_id = self.file_id_list[0]
        file_id = 'p001_041'
        # p001_041, B=389
        self.logger.info('Processing %s' % file_id)
        feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
        feed_dict, batch_size = self.make_feed_dict_B1(feed_dict, 20)
        print('-'*10)
        # lambda_SBD = self.model.gen_lambda_SBD_value(feed_dict)

        x_dict, y_dict = self.model.numpy_to_tensor(feed_dict)
        layer_list = self.model.nn_model.layer_list
        x_dict_2 = layer_list[1](layer_list[0](x_dict))
        # x_dict_2 contains f_SBM, tau_SBM, vuv_SBM, wav_SBMT
        # torch.Size([1, 1, 12]), torch.Size([1, 1, 12]), torch.Size([1, 1, 12]), torch.Size([1, 1, 12, 240])
        x = x_dict_2['wav_SBMT']
        f = x_dict_2['f_SBM']
        tau = x_dict_2['tau_SBM']
        sinenet_layer = layer_list[2].layer_fn
        sin_cos_matrix = sinenet_layer.sinenet_fn.construct_w_sin_cos_matrix(f,tau)
        print(sin_cos_matrix)
        print(sin_cos_matrix.size())
        # torch.Size([1, 1, 12, 128, 240])




        




    def run_2(self):
        # Plot window-level waveform, f, tau, vuv
        file_id = self.file_id_list[0]
        self.logger.info('Processing %s' % file_id)
        feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
        # print(feed_dict)
        # p001_041, B=389
        h = feed_dict['h'][0,20]
        print(h.shape) # 3036
        w = h[:3000]
        f = h[3000:3012]
        t = h[3012:3024]
        v = h[3024:3036]
        self.plot_wav(wav_data=w,tau=t)

    def plot_wav(self, wav_data=None, f=None, tau=None, v=None):
        x_list = []
        y_list = [] 
        l_list = [] # names

        if wav_data is not None:
            y_list.append(wav_data)
            x_list.append(numpy.arange(wav_data.shape[0])+1)
            l_list.append('wav')

        if f is not None:
            m_f = numpy.mean(f)
            y_list.append(f/m)
            x_list.append(numpy.arange(f.shape[0])*self.M_stride+self.T_M/2)
            l_list.append('f mean %.3f' % m_f)

        if tau is not None:
            print(tau)
            tau_1 = tau * self.wav_sr + numpy.arange(tau.shape[0]) * self.M_stride
            print(tau_1)
            x_list.append(tau_1)
            y_list.append([0]*tau.shape[0])
            l_list.append('pitch locations')


        if v is not None:
            y_list.append(v)
            x_list.append(numpy.arange(v.shape[0])*self.M_stride+self.T_M/2)
            l_list.append('vuv')

        # fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/wav_pitches_filters.png'
        # self.graph_plotter.single_plot(fig_file_name, x_list, y_list,l_list,title='Many things', x_label='Time/Samples', y_label='')
        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/wav_pitches.png'
        self.graph_plotter.one_line_one_scatter(fig_file_name, x_list, y_list,l_list,title='Waveform and pitch locations', x_label='Time/Samples', y_label='')
        

    def run_1(self):
        # Debugs
        file_id = self.file_id_list[0]
        self.logger.info('Processing %s' % file_id)
        input_list, B = self.compute_distance_list(file_id)
        print(input_list)

    def plot_cosine_distance(self, input_list, input_name):
        # Store old results for comparison
        l_list = [] # loss
        m_list = [] # model name

        l_list.append(input_list)
        m_list.append(input_name)

        l_list.append([0.005038160760272445, 0.007097771609965344, 0.008655920002770831, 0.009631910386890293, 0.01054470438975244, 0.011861003976885633, 0.012713511578595953, 0.013449126591101509, 0.014125597334154373, 0.014990042010347465, 0.01556940802817315, 0.015926362094456768, 0.016512530020217332, 0.01726089170628988, 0.017988835807296353, 0.018286953617942157, 0.01887106654493233, 0.018931773611309193, 0.01986698489471506, 0.02042461126853259, 0.02097163656208203, 0.02149595184567788, 0.02188159540582682, 0.022166288410400387, 0.022608406938822174, 0.022732770785779528, 0.022709346507042972, 0.02338117388570781, 0.023698947963337564, 0.02421990474669619, 0.02470378957186105, 0.024990410801061225, 0.025237119228254105, 0.025846972651342358, 0.025930172451062647, 0.026194684433474075, 0.02632351396972189, 0.026789127627095784, 0.027046256078929524, 0.027351572642521988, 0.027557748864613582, 0.027904626667476867, 0.02813160941114705, 0.028320488991445843, 0.02850281463163438, 0.028735891130648682, 0.028912767991981003, 0.029246931710620035, 0.029462010050146772, 0.028653953089282178])
        m_list.append('vocoder')

        l_list.append([0.00341437362281787, 0.007673577785851113, 0.01068440415983571, 0.013625429189254974, 0.014773644097680378, 0.014146327376177688, 0.013341482840406862, 0.011357867253929087, 0.009734411321450129, 0.012243781295739919, 0.015188466832288553, 0.017098506108083313, 0.01893121193212525, 0.019269892957405853, 0.018120155005434523, 0.01684741846766237, 0.01453048873670196, 0.012575948276029713, 0.014464953139654747, 0.016729227833746384, 0.017994181957260134, 0.019172409243883432, 0.018825121374635746, 0.01695241102497672, 0.015023886561205764, 0.012042021274882322, 0.00973445744790355, 0.012491735729867923, 0.01593438998726956, 0.018363114201093777, 0.02076337364907009, 0.02165860128220515, 0.021068385777797537, 0.02034086719315672, 0.018613437869634618, 0.0172124498244901, 0.019372343679141923, 0.02190002990806287, 0.023524476121135045, 0.025086180315711956, 0.02533514395629323, 0.024275548744180035, 0.02312834708178155, 0.021041946802907607, 0.019302258245963228, 0.02103025574826556, 0.023077710317538008, 0.024213043766928177, 0.02525568625831518, 0.024890441804073883])
        m_list.append('wav_SincNet')

        l_list.append([0.004797102939699798, 0.01073969577700829, 0.013370142094256823, 0.015390508561948386, 0.01863291514306373, 0.02110429473546159, 0.0236619892876271, 0.02652389174902997, 0.028882466658024235, 0.03140379285227267, 0.0337841305123872, 0.035366727387081236, 0.03713846898836034, 0.039147872972376055, 0.04074964860846563, 0.04247530462331547, 0.0443411584184171, 0.04574381775323092, 0.04717711176720048, 0.04872312935797257, 0.04985663657348687, 0.05108854897388955, 0.05249561896751781, 0.053526534333478405, 0.05464298647914418, 0.0558685195792632, 0.056832008266039846, 0.057787348096735254, 0.05882470617265783, 0.05965834662859121, 0.060614568273334855, 0.06159736907559605, 0.06238765927125395, 0.06323231699327952, 0.06418699338874531, 0.06488319444137937, 0.06566413323042258, 0.06647003960462661, 0.06714861233252227, 0.06787482329452738, 0.06866556522087205, 0.06930296636615868, 0.06992807069470938, 0.07056076124909097, 0.07118747773730018, 0.07179313971512863, 0.07246864263328258, 0.07309873903212702, 0.07372507656493084, 0.07431622455061265])
        m_list.append('wav_SineNet_v0')

        l_list.append([0.009968847644946464, 0.015137879285658544, 0.016603846558799262, 0.017986597770596028, 0.02135260245122312, 0.024003340627838172, 0.02594298578138163, 0.029021886662590368, 0.031047850701758715, 0.032953425700983814, 0.034547767824657724, 0.03656452635171971, 0.03832582392901891, 0.04044571634797096, 0.04131339988004536, 0.04217500524206582, 0.04475920018841279, 0.04580392559927424, 0.04715644023847746, 0.0490396249053457, 0.048848936586099456, 0.04994339945809216, 0.05205207529559497, 0.05346110883908661, 0.05386827804541797, 0.05556106142150225, 0.05676314770435762, 0.057171674613831415, 0.05824583296626321, 0.05981447351985242, 0.06048810195807768, 0.06111304182197334, 0.06221112031783271, 0.06294639229776339, 0.06428165998861257, 0.064631669687336, 0.06486214914140674, 0.06635167000367859, 0.06706393054501598, 0.06736326125023756, 0.06768769957337337, 0.06818034766229786, 0.06975488824451809, 0.07042314297499912, 0.07084993019295868, 0.0713198897947984, 0.07230802606820337, 0.07303154684333468, 0.0745160736771612, 0.07477174982001299])
        m_list.append('wav_SineNet_v2')

        x_list = []
        y_list = []
        for l in l_list:
            x = numpy.arange(len(l))+1
            x_list.append(x)
            y_list.append(l)

        fig_file_name = '/home/dawna/tts/mw545/Export_Temp/PNG_out/pos_test_cosine_loss.png'
        self.graph_plotter.single_plot(fig_file_name, x_list, y_list,m_list,title='Distance against Sample Shift', x_label='Sample Shift', y_label='Cosine Distance')




    def compute_distance_list(self, file_id, distance_type='cosine'):
        distance_list = numpy.zeros(self.max_distance)
        speaker_id = file_id.split('_')[0]
        feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[0])
        if distance_type=='cosine':
            lambda_SBD_0 = self.model.gen_lambda_SBD_value(feed_dict)
        elif distance_type=='KL':
            p_SD_0 = self.model.gen_p_SD_value(feed_dict)

        for i in range(0, self.max_distance):
            feed_dict, batch_size = self.data_loader.make_feed_dict(file_id_list=[file_id],start_sample_list=[i+1])
            if distance_type=='cosine':
                lambda_SBD_i = self.model.gen_lambda_SBD_value(feed_dict)
                distance_list[i] = self.dv_calculator.compute_SBD_distance(lambda_SBD_0, lambda_SBD_i, 'cosine') # This is sum, not mean
                B = batch_size
            elif distance_type=='KL':
                p_SD_i = self.model.gen_p_SD_value(feed_dict)
                distance_list[i] = self.dv_calculator.compute_SD_distance(p_SD_0, p_SD_i, 'entropy') # This is sum, not mean
                B = 1

        return distance_list, B





def run_sinenet_f_tau_test(cfg):
    from exp_mw545.exp_dv_wav_sinenet_v2 import dv_y_wav_sinenet_configuration
    dv_y_cfg = dv_y_wav_sinenet_configuration(cfg, cache_files=False)
    dv_y_cfg.input_data_dim['B_stride'] = int( dv_y_cfg.cfg.wav_sr/200)   # Change stride at inference time
    dv_y_cfg.update_wav_dim()
    test_class = Build_SineNet_F_Tau_Test(cfg, dv_y_cfg)
    test_class.run()







