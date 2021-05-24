DATA_PATH=TOF-master/data/cross-lingual/continual_train
CODE_PATH=TOF-master/code/cross-lingual/continual
CKPT_PATH=TOF-master/ckpts/es
BERT_PATH=multi_cased_L-12_H-768_A-12
BERT_TOKENIZER=multi_cased_L-12_H-768_A-12/vocab.txt

# merge the MRC dataset in the following path:
# 1. MRC-src: TOF-master/data/cross-lingual/init_data/MRC/MRC-src/squad_en/sample_10000/mrc-ner.train
# 2. MRC-tgt: TOF-master/data/cross-lingual/init_data/MRC/MRC-tgt/esp_qa/mrc-ner.train
# 3. MRC-trans: TOF-master/data/cross-lingual/init_data/MRC/MRC-trans/squad_es/mrc-ner.train
# 4. MRC-ner: TOF-master/data/cross-lingual/init_data/MRC/NER-MRC/conll_mrc_en/mrc-ner.train
# 5. MRC-ner_trans: TOF-master/data/cross-lingual/init_data/MRC/NER-MRC-trans/conll_mrc_es/mrc-ner.train
# 6. MRC-pseudo: TOF-master/data/cross-lingual/continual_train/MRC/esp/mrc-ner.train 
# dev : only 1 & 2 & 4
# target data path: ${DATA_PATH}/${mrc_data_mode}

mrc_data_mode="tgt+src+trans+ner+ner_trans+pseudo"
CUDA_VISIBLE_DEVICES=0  python3 $CODE_PATH/mrc-tuning/train_bert_mrc.py \
    --optimizer_type "adamw" \
    --lr_scheduler_type "ladder" \
    --lr_min 8e-6 \
    --loss_type "ce" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --data_dir "${DATA_PATH}/${mrc_data_mode}" \
    --data_mode "tgt+src+trans+conll+conll_trans+pseudo" \
    --lang_type "esp" \
    --n_gpu 1 \
    --entity_sign "flat" \
    --num_data_processor 20 \
    --data_sign "conll03" \
    --logfile_name "log.txt" \
    --bert_tokenizer "$BERT_TOKENIZER" \
    --bert_model "path_to_best_checkpoint_in_es_pseudo_train_ner_output"\
    --config_path "multi_bert_config.json" \
    --output_dir "$CKPT_PATH/es_continual_train_mrc_output" \
    --dropout 0.1 \
    --checkpoint 600 \
    --max_seq_length 160 \
    --train_batch_size 16 \
    --dev_batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 5e-5\
    --weight_start 1.0 \
    --weight_end 1.0 \
    --weight_span 1.0 \
    --entity_threshold 0.15 \
    --num_train_epochs 6 \
    --seed 2333 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --only_train

task_data_mode="tgt+src+pseudo"
CUDA_VISIBLE_DEVICES=0 python -W ignore $CODE_PATH/task-tuning.py \
    --data_dir="$DATA_PATH" \
    --lang_type="esp" \
    --data_mode=$task_data_mode \
    --bert_tokenizer="$BERT_TOKENIZER" \
    --bert_model="$BERT_PATH" \
    --output_dir="$CKPT_PATH/es_continual_train_ner_output" \
    --trained_model_dir="$CKPT_PATH/es_continual_train_mrc_output" \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --do_test \
    --train_batch_size=64 \
    --learning_rate=5e-5 \
    --num_train_epochs=6 \
    --warmup_proportion=0.1 \
    --seed=2019
