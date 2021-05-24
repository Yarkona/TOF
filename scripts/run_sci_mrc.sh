# MRC for 16G P100
DATA_PATH=TOF-master/data/cross-domain/scitech_ner/mrc_train
CODE_PATH=TOF-master/code/cross-domain
# CKPT_PATH=path_to_save_your_checkpoints
CKPT_path=TOF-master/ckpts/sci
BERT_TOKENIZER=bert-base-cased-vocab.txt
BERT_PATH=bert-base-cased.tar.gz

CUDA_VISIBLE_DEVICES=0 python -W ignore $CODE_PATH/domain-tuning.py \
    --data_dir="$DATA_PATH" \
    --bert_model="$BERT_PATH" \
    --output_dir="$DATA_PATH/lm_output_sci/" \
    --max_seq_length=128 \
    --do_train \
    --train_batch_size=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=3 \
    --warmup_proportion=0.1 \
    --seed=2019
# merge the MRC dataset in the following path:
# 1. MRC-src: TOF-master/data/cross-lingual/init_data/MRC/MRC-src/squad_en/sample_10000/mrc-ner.train
# 2. MRC-tgt: TOF-master/data/cross-domain/news_qa/mrc-ner.train
# 3. MRC-pseudo: TOF-master/data/cross-domain/scitech_ner/continual_train/mrc/mrc-ner.train
# dev : only 1 & 2
# target data path: ${DATA_PATH}/${mrc_data_mode}
mrc_data_mode="src_tgt_ner"

CUDA_VISIBLE_DEVICES=0  python3 $CODE_PATH/mrc-tuning/train_bert_mrc.py \
    --optimizer_type "adamw" \
    --lr_scheduler_type "ladder" \
    --lr_min 8e-6 \
    --loss_type "ce" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --data_dir "$DATA_PATH/${mrc_data_mode}" \
    --n_gpu 1 \
    --entity_sign "flat" \
    --num_data_processor 10 \
    --data_sign "conll03" \
    --logfile_name "log.txt" \
    --bert_tokenizer "$BERT_TOKENIZER" \
    --bert_model "$DATA_PATH/lm_output_sci/" \
    --config_path "en_bert_base_cased.json" \
    --output_dir "$CKPT_PATH/sci_mrc_train_mrc_output" \
    --dropout 0.1 \
    --checkpoint 600 \
    --max_seq_length 160 \
    --train_batch_size 16 \
    --dev_batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 2e-5 \
    --weight_start 1.0 \
    --weight_end 1.0 \
    --weight_span 1.0 \
    --entity_threshold 0.15 \
    --num_train_epochs 6 \
    --seed 2019 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1 \
    --only_eval_dev
CUDA_VISIBLE_DEVICES=0 python -W ignore $CODE_PATH/task-tuning.py \
    --data_dir="$DATA_PATH" \
    --bert_tokenizer="$BERT_TOKENIZER" \
    --bert_model="$BERT_PATH" \
    --output_dir="$CKPT_PATH/sci_mrc_train_ner_output" \
    --trained_model_dir="$CKPT_PATH/sci_mrc_train_mrc_output" \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --do_test \
    --train_batch_size=32 \
    --learning_rate=5e-5 \
    --num_train_epochs=6 \
    --warmup_proportion=0.1 \
    --seed=2019
