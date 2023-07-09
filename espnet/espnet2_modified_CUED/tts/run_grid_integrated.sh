echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export CUDA_VISIBLE_DEVICES=0
#echo $LD_LIBRARY_PATH

cd /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated

# spk_model_name=cmp
# train_spk_dataset_type=cmp_binary_86_40
# inference_spk_dataset_type=cmp_binary_86_40

# spk_model_name=sincnet
# train_spk_dataset_type=wav_binary_3000_3000
# inference_spk_dataset_type=wav_binary_3000_120
# init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet_concat_dynamic_5_seconds/valid.loss.best.pth

# spk_model_name=sincnet_4800
# train_spk_dataset_type=wav_binary_4800_4800
# inference_spk_dataset_type=wav_binary_4800_120

# spk_model_name=sinenet
# train_spk_dataset_type=wav_f_tau_vuv_binary_3000_3000
# inference_spk_dataset_type=wav_f_tau_vuv_binary_3000_120
# init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_frame_dynamic_5_seconds/valid.loss.best.pth

# spk_model_name=sinenet_v1
# train_spk_dataset_type=wav_f_tau_vuv_binary_3000_3000
# inference_spk_dataset_type=wav_f_tau_vuv_binary_3000_120
# init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_v1_frame_dynamic_5_seconds/valid.loss.best.pth

# spk_model_name=sinenet_v2
# train_spk_dataset_type=wav_f_tau_vuv_binary_3000_3000
# inference_spk_dataset_type=wav_f_tau_vuv_binary_3000_120
# init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_v2_frame_dynamic_5_seconds/valid.loss.best.pth

# spk_model_name=sinenet_4800
# train_spk_dataset_type=wav_f_tau_vuv_binary_4800_4800
# inference_spk_dataset_type=wav_f_tau_vuv_binary_4800_120

# spk_model_name=gst
# train_spk_dataset_type=cmp_binary_86_1
# inference_spk_dataset_type=cmp_binary_86_1

# spk_model_name=cmp_lab
# train_spk_dataset_type=cmp_lab_binary_86_40_5
# inference_spk_dataset_type=cmp_lab_binary_86_40_5
# init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_lab_5s_dynamic_5_seconds/valid.loss.best.pth

#spk_model_name=sincnet_lab
#train_spk_dataset_type=wav_lab_binary_3000_3000_5
#inference_spk_dataset_type=wav_lab_binary_3000_120_5
#init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sincnet_lab_5s_dynamic_5_seconds/valid.loss.best.pth

spk_model_name=sinenet_v2_lab_lr3
train_spk_dataset_type=wav_f_tau_vuv_lab_binary_3000_3000_5
inference_spk_dataset_type=wav_f_tau_vuv_lab_binary_3000_120_5
init_param=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage/exp/tts_train_raw_phn_tacotron_g2p_en_no_space_sinenet_v2_lab_5s_dynamic_5_seconds/valid.loss.best.pth

train_spk_dataset_name=dynamic_5_seconds


# Training
if true; then
    # ./run.sh --stage 5 --stop-stage 5 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --train_spk_dataset_name ${train_spk_dataset_name} --train_spk_dataset_type ${train_spk_dataset_type} --nj 4
    ./run.sh --stage 6 --stop-stage 6 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --train_spk_dataset_name ${train_spk_dataset_name} --train_spk_dataset_type ${train_spk_dataset_type} --init_param ${init_param}
fi

# train_spk_dataset_name=dynamic_5_seconds
test_sets=eval1
inference_model=valid.loss.best

# Multi-second test
if false; then
# for num_seconds in 5 10 15 20 30 40 50; do
for num_seconds in 5 30; do
    for num_draw in {1..30}; do
    # for num_draw in 1; do
        inference_spk_dataset_name=same_${num_seconds}_seconds_per_speaker_draw_${num_draw}
        ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing true --inference_tag tf --generate_wav false --inference_keep_feats false --gpu_inference false --inference_nj 10
    done
    log_dir=exp/tts_train_${spk_model_name}_tacotron2_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}_${train_spk_dataset_name}/tf
    . ./path.sh && python3 ./pyscripts/pyscript_utils/compute_mean_std_num_secs.py ${log_dir} ${num_seconds}
done          
fi

# Speaker embedding generation
test_sets=spk_embed_eval1
if false; then
    inference_spk_dataset_name=speaker_draw_1_30_file_per_speaker_draw_30
    ./run.sh --stage 8 --stop-stage 8 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type} 

fi

train_spk_dataset_name=${train_spk_dataset_name}
inference_spk_dataset_name=same_30_seconds_per_speaker_draw_1
# Waveform Generation
test_sets=eval1
if true; then
    ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing false --generate_wav false --gpu_inference false --inference_nj 10
    dataset_dir=exp/tts_train_${spk_model_name}_tacotron2_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}_${train_spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
    checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
    . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
fi

# Evaluate waveform
if true; then
    dataset_dir=exp/tts_train_${spk_model_name}_tacotron2_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}_${train_spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
    gen_wav_dir=${dataset_dir}/wav_pwg
    gt_wav_dir=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav_ref
    echo Testing ${gen_wav_dir}
    . ./path.sh
    python3 ./pyscripts/pyscript_utils/evaluate_mcd.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
    python3 ./pyscripts/pyscript_utils/evaluate_f0.py  ${gen_wav_dir}  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
    python3 ./pyscripts/pyscript_utils/evaluate_duration.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
fi



echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"


#####################################

# Multi-file test
if false; then
for num_adapt_sentence in {17..30}; do
    for num_draw in {1..30}; do
        inference_spk_dataset_name=same_${num_adapt_sentence}_file_per_speaker_draw_${num_draw}
        ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing true --generate_wav false
    done
done          
fi

