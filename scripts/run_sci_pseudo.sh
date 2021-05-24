DATA_PATH=TOF-master/data/cross-domain/scitech_ner/pseudo_train
CODE_PATH=TOF-master/code/cross-domain
# CKPT_PATH=path_to_save_your_checkpoints
CKPT_path=TOF-master/ckpts/sci
BERT_TOKENIZER=bert-base-cased-vocab.txt
BERT_PATH=bert-base-cased.tar.gz

CUDA_VISIBLE_DEVICES=0,1 python -W ignore $REPO_PATH/task-pseudo-tuning.py \
                --data_dir="$DATA_PATH" \
                --bert_tokenizer="$BERT_TOKENIZER" \
                --bert_model="$BERT_PATH" \
                --output_dir="$CKPT_PATH/sci_pseudo_train_ner_output" \
                --trained_model_dir="path_to_best_checkpoint_in_sci_mrc_train_ner_output"\
                --max_seq_length=128 \
                --do_train \
                --do_eval \
                --do_test \
                --train_batch_size=128 \
                --learning_rate=8e-5 \
                --num_train_epochs=6 \
                --warmup_proportion=0.1 \
                --seed=2019
