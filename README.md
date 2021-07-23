# Target-oriented Fine-tuning for Zero-resource Named Entity Recognition

This repository contains the code of the recent research advances in Paper 
*Target-Oriented Fine-tuning for Zero-resource Named Entity Recognition*. 

Ying Zhang, Fandong Meng, Yufeng Chen, Jian Xu and Jie Zhou in Findings of ACL 2021.

If you find this repo helpful, please cite the following:
`@misc{zhang2021targetoriented,
      title={Target-Oriented Fine-tuning for Zero-Resource Named Entity Recognition}, 
      author={Ying Zhang and Fandong Meng and Yufeng Chen and Jinan Xu and Jie Zhou},
      year={2021},
      eprint={2107.10523},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}`

For any question, please feel free to post Github issues.

## Install Requirements
We build our project on pytorch==1.2 and pytorch_pretrained_bert==0.6.1. The most of source code are based on [AdaptaBERT](https://github.com/xhan77/AdaptaBERT) and [mrc-for-flat-nested-ner](https://github.com/ShannonAI/mrc-for-flat-nested-ner/tree/master).

## Data Preparation

   1. Enter the data directory

      `cd ./data/`

   2. cross-domain

      In cross-domain scenarios, we take the conll03_English as the source NER data (both the training and development set). We take the test set from three benckmarks: sciTech, twitter_NER, and WNUT16. For MRC data, we consider news_qa for sciTech and tweet_qa for both twitter_NER and WNUT16. We take twitter_NER for example to introduce the whole process of data preprocess and explotation. All raw data and preprocessed data are provided in the file.

      1. NER data

         1. Enter the twitter_ner directory

            `cd ./cross-domain/twitter_ner `

         2. **mrc_train：** We use python file raw2bio.py to transform the raw data twitter_ner/*.txt into BIO schema mrc_train/*.bio. 

            `python ./raw2bio.py dev.txt mrc_train/dev.bio`

            Based on BIO files, we generate the training data mrc_train/*.pkl with process_twitter.py.

            `python ./process_twitter.py `

         3. **pseudo_train:** we generate pseudo file twitter_train/twitter.*.bio with the best checkpoint of mrc_train on the training and development set of unlabeled raw data.

            `bash ./generate_pseudo.sh` 

            Then we generate the *.pkl with process_twitter.py. Note that you need to modify path of BIO file.

            `python ./process_twitter.py`

         4. **continual_train:** first we generate new pseudo data with the fine-tuned NER model from pseudo_train.

            `bash ./generate_pseudo.sh`

            Then transform the pseudo NER data into MRC format with the code and scripts in the continual_train/mrc following (li et al., 2020).

            `bash ./continual_train/mrc/gen_mrc_ner_dataset.sh`

      2. MRC data

         1. Enter the directory

            `cd ./cross-domain/tweet_qa`
            
         2. The raw data of tweet_qa is tweet_qa/*.json. We use the python scripts preprocess_mrc.py to transform  raw data into the unified MRC format. For example,

            `python ./preprocess_mrc.py train.json mrc-ner.json`

   3. cross-lingual

      1. **Raw Data：** we collect four kinds of NER data, where CoNLL03 English (en) is regarded as source, CoNLL03 German (de), and CoNLL02 Spanish (es), and CoNLL02 Dutch(nl) as target. MRC datasets for different target langauges are MLQA(es) and XQuAD(de). Dutch MRC is unavailable.

      2. **Preprocessed Data：** We consider two kinds of NER data and five kinds of MRC data.

         | Task Type     | Data Source                                 | Data Path                   |
         | ------------- | ------------------------------------------- | --------------------------- |
         | NER           | conll03_en                                  | init_data/NER/              |
         | NER-trans     | conll03_en_trans {es, de, nl}               | init_data/NER/trans         |
         | MRC-tgt       | esp_qa, deu_qa                              | init_data/MRC/MRC-tgt       |
         | MRC-src       | SQuAD_en                                    | init_data/MRC/MRC-src       |
         | MRC-src-trans | SQuAD_en_trans {es, de, nl}                 | init_data/MRC/MRC-trans     |
         | NER-MRC       | conll03_en in MRC format                    | init_data/MRC/NER-MRC       |
         | NER-MRC-trans | conll03_en_trans {es, de, nl} in MRC format | init_data/MRC/NER-MRC-trans |

         **Translation Generation** is implemented with MUSE, both code and scripts are provided in the directory ./translate/cross-lingual_NER-master.

         Translate the NER data with the command

         `bash run_transfer_training_data_bio.sh`

         Translate the MRC data with the command

         `bahs run_transfer_mrc_data.sh`

      3. **Training Data：**

         1. Enter the directory

            `cd ./cross-lingual`

         2. **MRC_train:** this step uses all preprocessed data as the inputs. 

            `bash run_prepro.sh # in mrc_train` 

         3. **Pseudo_train:** this step generates pseudo labels on the training set in target languages {es, de, nl} with the best checkpoint of NER model from MRC_train. We continue to fine-tune the best NER checkpoint on the pseudo labeled data.

            `bash run_prepro.sh # in pseudo_train` 

            Note that we have provided the pseudo labeled file in pseudo_train, e.g., <esp/deu/ned>.<train/dev>.bio. If not, you can generate it with the command as follows based on your best checkpoint:

            `bash generate_pseudo.sh`

         4. **Continual_train:** this step generates pseudo labels on the training set in target langauges {es, de,nl} with the best checkpoint of NER model from Pseudo_train. 

            `bash run_prepro.sh # in continual_train`
         
            Then pseudo labeled NER data is transformed into the MRC format. We continual to fine-tune both MRC and NER model with pseudo data.
         
            `bash ./continual_train/MRC/gen_mrc_ner_datasets.sh`

## Training

   1. Enter into the directory of scripts:

      `cd ./scripts`

   2. **MRC-enhancing:** First merge the MRC data as the scripts in run_sci_mrc.sh, and then conduct the following command.

      `bash run_sci_mrc.sh`

   3. **Pseudo-enhancing**: First generate the pseudo data with the best checkpoint of MRC-enhancing NER model wit h the command in the /data. Then perform the command:

      `bash run_sci_pseudo.sh`

   4. **Continual-enhancing:** First generate the pseudo data  with the best checkpoint of Pseudo-enhancing NER models and transform it into MRC format. Then prepare the MRC data as the scripts in run_sci_continual.sh. Finally perform the command:

      bash run_sci_continual.sh

## Testing: 
   evaluate the checkpoint with command eval_ckpt.sh

   `cd ./scripts`

   `bash eval_ckpt.sh`

