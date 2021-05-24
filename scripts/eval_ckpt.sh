# MRC for 16G P100
REPO_PATH=TOF-master
BERT_TOKENIZER=bert-base-cased-vocab.txt
BERT_PATH=bert-base-cased.tar.gz

CUDA_VISIBLE_DEVICES=0 python -W ignore $REPO_PATH/code/cross-domain/test.py \
     --data_dir="$REPO_PATH/data" \
     --bert_tokenizer="$BERT_TOKENIZER" \
     --bert_model="$BERT_PATH" \
     --output_dir="$REPO_PATH/error_analysis_output_bert_5e-5" \
     --trained_model_dir="$REPO_PATH/trained_model_bert_5e-5" \
     --max_seq_length=128 \
     --do_test \
     --eval_batch_size=1 \
     --seed=2019
perl conlleval.txt < output.system
