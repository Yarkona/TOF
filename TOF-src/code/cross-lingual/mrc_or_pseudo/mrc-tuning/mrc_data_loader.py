#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 
# description:
# 

import os
import math
import torch
import random
from glob import glob
from multiprocessing import Pool
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from mrc_utils import convert_examples_to_features



class MRCNERDataLoader(object):
    def __init__(self, config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True, entity_scheme="bes"):

        self.data_ratio = config.data_ratio
        self.data_dir = config.data_dir
        self.data_mode = config.data_mode
        self.lang_type = config.lang_type
        self.save_cache_path = os.path.join(self.data_dir, self.lang_type+"_"+self.data_mode)
        if not os.path.exists(self.save_cache_path):
            os.mkdir(self.save_cache_path)
        self.max_seq_length= config.max_seq_length
        self.entity_scheme = entity_scheme
        self.distributed_data_sampler = config.n_gpu > 1 and config.data_parallel == "ddp"

        if mode == "train":
            self.train_batch_size = config.train_batch_size
            self.dev_batch_size = config.dev_batch_size
            self.test_batch_size = config.test_batch_size
            self.num_train_epochs = config.num_train_epochs 
        elif mode == "test":
            self.test_batch_size = config.test_batch_size
        elif mode == "transform_binary_files":
            print("=*="*15)
            print("Transform pre-processed MRC-NER datasets into binary files. ")
            print("max_sequence_length is : ", config.max_seq_length)
            print("data_dir is : ", config.data_dir)
            print("=*="*15)
        else:
            raise ValueError("[mode] for MRCNERDataLoader does not exist.")

        self.data_processor = data_processor 
        self.label_list = label_list 
        self.allow_impossible = allow_impossible
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_seq_length 
        self.data_cache = config.data_cache

        self.num_train_instances = 0 
        self.num_dev_instances = 0 
        self.num_test_instances = 0

    def examples_for_diff_data_mode(self, data_sign):
        """
        lang_type = ["esp", "deu1", "deu2", "ned", "no"]
        data_mode = ["tgt", "src",  "src+trans", "tgt+trans", "tgt+src+trans"]
        """
        src_qa = "squad_en"
        src_ner_qa = "conll_mrc_en"
        if self.lang_type == "esp":
            tgt_qa = "esp_qa"
            src_trans_qa = "squad_es"
            src_trans_ner_qa = "conll_mrc_es"
            pseudo_qa = "esp_pseudo"
        elif self.lang_type == "deu1":
            tgt_qa = "deu_qa_1"
            src_trans_qa = "squad_de"
            src_trans_ner_qa = "conll_mrc_de"
            pseudo_qa = "deu_pseudo"
        elif self.lang_type == "deu2":
            tgt_qa = "deu_qa_2"
            src_trans_qa = "squad_de"
            src_trans_ner_qa = "conll_mrc_de"
            pseudo_qa = "deu_pseudo"
        elif self.lang_type == "ned":
            tgt_qa = None
            src_trans_qa = "squad_nl"
            src_trans_ner_qa = "conll_mrc_nl"
            pseudo_qa = "ned_pseudo"
        elif self.lang_type == "no":
            tgt_qa = None
            src_trans_qa = "squad_no"
            src_trans_ner_qa = "conll_mrc_no"
        else:
            print("Language type is not valid!")
        
        if self.data_mode == "tgt":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return 
        if self.data_mode == "src":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
            update_data_path = os.path.join(self.data_dir, src_qa)
        if self.data_mode == "src+trans":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
            src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
            limit_len = 10000
            sp_examples = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
            examples.extend(sp_examples)
            
        if self.data_mode == "tgt+src":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
                src_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
                limit_len = 10000
                sp_examples = random.sample(src_examples, min(limit_len, len(src_examples)))
                examples.extend(sp_examples)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return
        if self.data_mode == "tgt+src+trans":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
                src_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
                src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
                limit_len = 10000
                sp_examples_1 = random.sample(src_examples, min(limit_len, len(src_examples)))
                sp_examples_2 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
                examples.extend(sp_examples_1)
                examples.extend(sp_examples_2)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return
        
        if self.data_mode == "src+trans+conll+pseudo":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
            src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
            src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
            pseudo_examples = self.data_processor.get_examples(os.path.join(self.data_dir, pseudo_qa), data_sign)
            limit_len = 10000
            sp_examples_1 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
            sp_examples_2 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
            sp_examples_3 = random.sample(pseudo_examples, min(limit_len, len(pseudo_examples)))
            examples.extend(sp_examples_1)
            examples.extend(sp_examples_2)
            examples.extend(sp_examples_3)

        if self.data_mode == "src+trans+conll":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
            src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
            src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
            limit_len = 10000
            sp_examples_1 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
            sp_examples_2 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
            examples.extend(sp_examples_1)
            examples.extend(sp_examples_2)
 
        if self.data_mode == "trans+pseudo+conll_trans":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
            pseudo_examples = self.data_processor.get_examples(os.path.join(self.data_dir, pseudo_qa), data_sign)
            src_ner_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_ner_qa), data_sign)
            limit_len = 10000
            sp_examples_1 = random.sample(pseudo_examples, min(limit_len, len(pseudo_examples)))
            sp_examples_2 = random.sample(src_ner_trans_examples, min(limit_len, len(src_ner_trans_examples)))
            examples.extend(sp_examples_1)
            examples.extend(sp_examples_2)
        if self.data_mode == "src+trans+conll+conll_trans":
            examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
            src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
            src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
            src_ner_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_ner_qa), data_sign)
            limit_len = 10000
            sp_examples_1 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
            sp_examples_2 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
            sp_examples_3 = random.sample(src_ner_trans_examples, min(limit_len, len(src_ner_trans_examples)))
            examples.extend(sp_examples_1)
            examples.extend(sp_examples_2)
            examples.extend(sp_examples_3)

        if self.data_mode == "tgt+src+trans+conll":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
                src_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
                src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
                src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
                limit_len = 10000
                sp_examples_1 = random.sample(src_examples, min(limit_len, len(src_examples)))
                sp_examples_2 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
                sp_examples_3 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
                examples.extend(sp_examples_1)
                examples.extend(sp_examples_2)
                examples.extend(sp_examples_3)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return
        if self.data_mode == "tgt+src+trans+conll+conll_trans+pseudo":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
                src_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
                src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
                src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
                src_ner_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_ner_qa), data_sign)
                pseudo_examples = self.data_processor.get_examples(os.path.join(self.data_dir, pseudo_qa), data_sign)
                limit_len = 10000
                sp_examples_1 = random.sample(src_examples, min(limit_len, len(src_examples)))
                sp_examples_2 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
                sp_examples_3 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
                sp_examples_4 = random.sample(src_ner_trans_examples, min(limit_len, len(src_ner_trans_examples)))
                sp_examples_5 = random.sample(pseudo_examples, min(limit_len, len(pseudo_examples)))
                examples.extend(sp_examples_1)
                examples.extend(sp_examples_2)
                examples.extend(sp_examples_3)
                examples.extend(sp_examples_4)
                examples.extend(sp_examples_5)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return
        if self.data_mode == "tgt+src+trans+conll+conll_trans":
            if tgt_qa is not None:
                examples = self.data_processor.get_examples(os.path.join(self.data_dir, tgt_qa), data_sign)
                src_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_qa), data_sign)
                src_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_qa), data_sign)
                src_ner_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_ner_qa), data_sign)
                src_ner_trans_examples = self.data_processor.get_examples(os.path.join(self.data_dir, src_trans_ner_qa), data_sign)
                limit_len = 10000
                sp_examples_1 = random.sample(src_examples, min(limit_len, len(src_examples)))
                sp_examples_2 = random.sample(src_trans_examples, min(limit_len, len(src_trans_examples)))
                sp_examples_3 = random.sample(src_ner_examples, min(limit_len, len(src_ner_examples)))
                sp_examples_4 = random.sample(src_ner_trans_examples, min(limit_len, len(src_ner_trans_examples)))
                examples.extend(sp_examples_1)
                examples.extend(sp_examples_2)
                examples.extend(sp_examples_3)
                examples.extend(sp_examples_4)
            else:
                print("No qa dataset for {}".format(self.lang_type))
                return


        sub_examples = random.sample(examples, round(len(examples)*self.data_ratio))
        print("data_size:{}, data_ratio:{}".format(len(sub_examples), self.data_ratio))        
        return sub_examples
            
        

    def convert_examples_to_features(self, data_sign="train", num_data_processor=1, logger=None):

        # logger.info("=*="*10)
        # logger.info(f"loading {data_sign} data ... ...")
        print(f"loading {data_sign} data ... ...")
        examples = self.examples_for_diff_data_mode(data_sign)
        if data_sign == "train":
            self.num_train_instances = len(examples)
        elif data_sign == "dev":
            self.num_dev_instances = len(examples)
        elif data_sign == "test":
            self.num_test_instances = len(examples)
        else:
            raise ValueError("please notice that the data_sign can only be train/dev/test !!")

        if num_data_processor == 1:
            cache_path = os.path.join(self.save_cache_path, "mrc-ner.{}.{}.cache.{}".format(self.data_ratio, data_sign, str(self.max_seq_len)))
            if os.path.exists(cache_path):
                # logger.info(f"%%%% %%%% Load Saved Cache files in {cache_path} %%% %%% ")
                features = torch.load(cache_path)
            else:
                features = convert_examples_to_features(examples, self.tokenizer, self.label_list, self.max_seq_length,
                                                    allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                torch.save(features, cache_path)
            return features

        def export_features_to_cache_file(idx, sliced_features, num_data_processor):
            cache_path = os.path.join(self.save_cache_path, "mrc-ner.{}.{}.cache.{}.{}-{}".format(self.data_ratio, data_sign, str(self.max_seq_len), str(num_data_processor), str(idx)))
            torch.save(sliced_features, cache_path)
            # logger.info(f">>> >>> >>> export sliced features to : {cache_path}")

        features_lst = []
        total_examples = len(examples)
        size_of_one_process = math.ceil(total_examples / num_data_processor)
        path_to_preprocessed_cache = os.path.join(self.save_cache_path, "mrc-ner.{}.{}.cache.{}.{}-*".format(self.data_ratio,data_sign, str(self.max_seq_len), str(num_data_processor)))
        collection_of_preprocessed_cache = glob(path_to_preprocessed_cache)

        if len(collection_of_preprocessed_cache) == num_data_processor:
            # logger.info(f"%%%% %%%% Load Saved Cache files in {self.data_dir} %%% %%% ")
            print(f"%%%% %%%% Load Saved Cache files in {self.save_cache_path} %%% %%% ")
        elif len(collection_of_preprocessed_cache) != 0:
            for item_of_preprocessed_cache in collection_of_preprocessed_cache:
                os.remove(item_of_preprocessed_cache)
            for idx in range(num_data_processor):
                start = size_of_one_process * idx
                end = (idx+1) * size_of_one_process if (idx+1)* size_of_one_process < total_examples else total_examples
                sliced_examples = examples[start:end]
                sliced_features = convert_examples_to_features(sliced_examples, self.tokenizer, self.label_list,
                                                               self.max_seq_length, allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                export_features_to_cache_file(idx, sliced_features, num_data_processor)
            del examples
        else:
            for idx in range(num_data_processor):
                start = size_of_one_process * idx
                end = (idx+1) * size_of_one_process if (idx+1)* size_of_one_process < total_examples else total_examples
                sliced_examples = examples[start:end]
                sliced_features = convert_examples_to_features(sliced_examples, self.tokenizer, self.label_list,
                                                               self.max_seq_length, allow_impossible=self.allow_impossible, entity_scheme=self.entity_scheme)
                export_features_to_cache_file(idx, sliced_features, num_data_processor)
            del examples

        multi_process_for_data = Pool(num_data_processor)
        for idx in range(num_data_processor):
            features_lst.append(multi_process_for_data.apply_async(MRCNERDataLoader.read_features_from_cache_file, args=(idx, self.data_ratio, self.save_cache_path, data_sign, self.max_seq_len, num_data_processor, logger)))

        multi_process_for_data.close()
        multi_process_for_data.join()
        features = []
        for feature_slice in features_lst:
            features.extend(feature_slice.get())

        # logger.info("check number of examples before and after data processing : ")
        # logger.info(f"{len(features)}, {total_examples}")
        # assert len(features) == total_examples

        return features

    def get_dataloader(self, data_sign="train", num_data_processor=1, logger=None):
        
        features = self.convert_examples_to_features(data_sign=data_sign, num_data_processor=num_data_processor, logger=logger)
        # logger.info(f"{len(features)} {data_sign} data loaded")
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        start_pos = torch.tensor([f.start_position for f in features], dtype=torch.long)   
        end_pos = torch.tensor([f.end_position for f in features], dtype=torch.long)
        span_pos = torch.tensor([f.span_position for f in features], dtype=torch.long)
        ner_cate = torch.tensor([f.ner_cate for f in features], dtype=torch.long)
        span_label_mask = torch.tensor([f.span_label_mask for f in features], dtype=torch.long)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, span_label_mask, ner_cate)
        
        if data_sign == "train":
            if self.distributed_data_sampler:
                # Please Note that DistributedSampler samples randomly
                datasampler = DistributedSampler(dataset)
            else:
                datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size)
        elif data_sign == "dev":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.dev_batch_size)
        elif data_sign == "test":
            datasampler = SequentialSampler(dataset) 
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size)

        return dataloader 

    @staticmethod
    def read_features_from_cache_file(idx, data_ratio, data_dir, data_sign, max_seq_len, num_data_processor, logger):
        cache_path = os.path.join(data_dir,
                                  "mrc-ner.{}.{}.cache.{}.{}-{}".format(data_ratio, data_sign, str(max_seq_len), str(num_data_processor), str(idx)))
        sliced_features = torch.load(cache_path)
        # logger.info(f"load sliced features from : {cache_path} <<< <<< <<<")
        return sliced_features

    def get_train_instance(self, ):
        return self.num_train_instances








