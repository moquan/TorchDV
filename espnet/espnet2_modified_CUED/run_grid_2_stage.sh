echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
# export CUDA_VISIBLE_DEVICES=1
# cd /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage
cd /data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_2_stage

# spk_embed_name=cmp_5s_concat
spk_embed_name=cmp_frame_concat
# spk_embed_name=sincnet_concat
# spk_embed_name=sinenet_v0_no_a
# spk_embed_name=sinenet_v1_5s
# spk_embed_name=sinenet_v1_frame
# spk_embed_name=sinenet_v2_frame
# spk_embed_name=cmp_lab_5s
# spk_embed_name=sincnet_lab_5s
# spk_embed_name=sinenet_v2_lab_5s
train_spk_dataset_name=dynamic_5_seconds
spk_embed_type=spk_embed_512

# Training
if false; then
    ./run.sh --stage 6 --stop-stage 6 --use_xvector true --train_config conf/train.yaml --spk_embed_name ${spk_embed_name} --train_spk_dataset_name ${train_spk_dataset_name} --spk_embed_type ${spk_embed_type}
fi

# Multiple seed training
if false; then
for seed in 1; do
    ./run.sh --stage 6 --stop-stage 6 --use_xvector true --train_config conf/train.yaml --spk_embed_name ${spk_embed_name} --train_spk_dataset_name ${train_spk_dataset_name} --spk_embed_type ${spk_embed_type} --train_args "--seed ${seed}"
done
fi

# spk_embed_name=sincnet_concat
# spk_embed_name=sinenet_frame
test_sets=eval1
inference_model=valid.loss.best
# inference_model=train.loss.best
train_spk_dataset_name=dynamic_5_seconds

# Waveform Generation
# train_spk_dataset_name=dynamic_15_seconds
test_sets=eval1
train_spk_dataset_name=dynamic_5_seconds
gt_wav_dir=/data/vectra2/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/VCTK-Corpus/wav_ref

# inference_spk_dataset_name=same_30_seconds_per_speaker_draw_1
# inference_spk_dataset_name=same_5_seconds_per_speaker_draw_1
if true; then
    # for spk_embed_name in cmp_frame ; do
    for seed in 0 1 2; do
    for inference_model in valid.loss.best train.loss.best; do
    for inference_spk_dataset_name in same_5_seconds_per_speaker_draw_1 same_30_seconds_per_speaker_draw_1; do
        ./run.sh --stage 7 --stop-stage 7 --use_xvector true --train_config conf/train.yaml   --spk_embed_name ${spk_embed_name}  --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name}  --spk_embed_type ${spk_embed_type} --inference_spk_dataset_name ${inference_spk_dataset_name}  --inference_use_teacher_forcing false --generate_wav false  --gpu_inference false --inference_nj 10 --train_args "--seed ${seed}"
        dataset_dir=exp/tts_train_raw_phn_tacotron_g2p_en_no_space_seed${seed}_${spk_embed_name}_${train_spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
        checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
        . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
        if true; then
        # Evaluate waveform
            dataset_dir=exp/tts_train_raw_phn_tacotron_g2p_en_no_space_${spk_embed_name}_${train_spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
            gen_wav_dir=${dataset_dir}/wav_pwg
            echo Testing ${gen_wav_dir}
            . ./path.sh
            python3 ./pyscripts/pyscript_utils/evaluate_mcd.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
            python3 ./pyscripts/pyscript_utils/evaluate_f0.py  ${gen_wav_dir}  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
            python3 ./pyscripts/pyscript_utils/evaluate_duration.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir}
        fi
    done
    done
    done
fi

if false; then
    # base_dir=exp/cmp_5_save/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_concat_dynamic_5_seconds/decode_valid.loss.best
    # for dir_name in same_30_seconds_per_speaker_draw_1/eval1 same_30_seconds_per_speaker_draw_1/eval_shared same_5_seconds_per_speaker_draw_1/eval1 same_5_seconds_per_speaker_draw_1/eval_shared; do
    base_dir=exp/cmp_5_save/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_5s_concat_dynamic_5_seconds/decode_train.loss.best
    base_dir=exp/tts_train_raw_phn_tacotron_g2p_en_no_space_cmp_frame_concat_dynamic_5_seconds/decode_valid.loss.best
    # for dir_name in same_30_seconds_per_speaker_draw_1/eval1 same_5_seconds_per_speaker_draw_1/eval1 ; do
    for dir_name in same_30_seconds_per_speaker_draw_1/eval1  ; do
        gen_wav_dir=${base_dir}/${dir_name}/wav_pwg
        echo Testing ${gen_wav_dir}
        . ./path.sh
        python3 ./pyscripts/pyscript_utils/evaluate_mcd.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
        python3 ./pyscripts/pyscript_utils/evaluate_f0.py  ${gen_wav_dir}  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
        python3 ./pyscripts/pyscript_utils/evaluate_duration.py  --wavdir ${gen_wav_dir} --gt_wavdir  ${gt_wav_dir} --outdir  ${gen_wav_dir} 
    done
fi





echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"



inference_tag=tf
# inference_model=train.loss.best
# inference_tag=tf_train
# inference_model=50epoch
# inference_tag=tf_50
# Multi-second test
if false; then
# for num_seconds in 5 10 15 20 30 40 50; do
for num_seconds in 5 30; do
    for num_draw in {1..30}; do
# for num_seconds in 30; do
#     for num_draw in {18..30}; do
        inference_spk_dataset_name=same_${num_seconds}_seconds_per_speaker_draw_${num_draw}
        ./run.sh --stage 7 --stop-stage 7  --use_xvector true --train_config conf/train.yaml --spk_embed_name ${spk_embed_name}  --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name}  --spk_embed_type ${spk_embed_type} --inference_spk_dataset_name ${inference_spk_dataset_name}  --inference_use_teacher_forcing true --inference_tag ${inference_tag} --generate_wav false --inference_keep_feats false # --train_args "--seed 1"
    done
    log_dir=exp/tts_train_raw_phn_tacotron_g2p_en_no_space_${spk_embed_name}_${train_spk_dataset_name}/tf
    . ./path.sh && python3 ./pyscripts/pyscript_utils/compute_mean_std_num_secs.py ${log_dir} ${num_seconds}
done          
fi