DATA_PATH=TOF-master/data/cross-lingual/pseudo_train
CODE_PATH=TOF-master/code/cross-lingual/mrc_or_pseudo
CKPT_PATH=TOF-master/ckpts/es
BERT_PATH=multi_cased_L-12_H-768_A-12
BERT_TOKENIZER=multi_cased_L-12_H-768_A-12/vocab.txt

CUDA_VISIBLE_DEVICES=0 python -W ignore $CODE_PATH/task-pseudo-tuning.py \
    --data_dir="$DATA_PATH" \
    --lang_type="esp" \
    --data_mode="tgt+src+pseudo" \
    --bert_tokenizer="$BERT_TOKENIZER" \
    --bert_model="$BERT_PATH" \
    --output_dir="$CKPT_PATH/es_pseudo_train_ner_output" \
    --trained_model_dir="path_to_best_checkpoint_in_es_mrc_train_ner_output" \
    --max_seq_length=128 \
    --do_train \
    --do_eval \
    --do_test \
    --train_batch_size=64 \
    --learning_rate=2e-5 \
    --num_train_epochs=6 \
    --warmup_proportion=0.1 \
    --seed=2019
