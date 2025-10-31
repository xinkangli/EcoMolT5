# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from typing import Optional
from model.GIMLET.data_utils import get_morgan_fingerprint
from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
from .graphormer_transform import graphormer_data_transform_tensor
from .graphormer_collator import collator_graph_data,padding
from .basic_collate import basic_collate
import time
# class CollatorForGIMLETLanguageModeling(DataCollatorForLanguageModeling):
# 
#     def __init__(self,**kwargs):
#         self.transform_in_collator= kwargs.pop('transform_in_collator')
#         self.rich_features = kwargs.pop('rich_features')
#         super().__init__(**kwargs)
# 
#     def __post_init__(self):
# 
#         if self.tf_experimental_compile:
#             import tensorflow as tf
# 
#             self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)
# 
#     def torch_call(self, examples):
#         graph_batch = []
#         text_batch = []
#         labels_batch = []
# 
#         # === 提取 SMILES 并生成 fingerprint ===
#         smiles_list = [str(example_data['graph']) for example_data in examples]  # graph 是 SMILES
#         fingerprints = get_morgan_fingerprint(smiles_list)  # 返回 FloatTensor (B, 2048)
#         # =====================================
# 
#         for idx, example_data in enumerate(examples):
#             graph_data = example_data['graph']
#             graph_batch.append(graph_data)
# 
#             text_batch.append({
#                 'input_ids': example_data['input_ids'],
#                 'attention_mask': example_data['attention_mask'],
#             })
# 
#             decoder_attention_mask = example_data.get('decoder_attention_mask')
# 
#             # 若为 None，则用与 labels 同 shape 的全 1 mask（或根据 labels != -100 生成更合理）
#             if decoder_attention_mask is None and example_data['labels'] is not None:
#                 decoder_attention_mask = (torch.tensor(example_data['labels']) != -100).long().tolist()
# 
#             labels_batch.append({
#                 'labels': example_data['labels'],
#                 'decoder_attention_mask': decoder_attention_mask
#             })
# 
#         # 图数据 + 文本 Padding
#         graph_batch = collator_graph_data(
#             graph_batch,
#             transform_in_collator=self.transform_in_collator,
#             rich_features=self.rich_features,
#         )
#         text_batch = padding(
#             text_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
#             pad_to_multiple_of=self.pad_to_multiple_of
#         )
#         labels_batch = padding(
#             labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
#             pad_to_multiple_of=self.pad_to_multiple_of
#         )
# 
#         # 构造最终 Batch
#         batch = {
#             'graph': graph_batch,
#             'input_ids': text_batch.data['input_ids'],
#             'attention_mask': text_batch.data['attention_mask'],
#             'labels': labels_batch.data['labels'],
#             'fingerprint': fingerprints,  # ✅ 加入生成的 fingerprint
#         }
#         if 'decoder_attention_mask' in labels_batch.data:
#             batch['decoder_attention_mask'] = labels_batch.data['decoder_attention_mask']
# 
#         return batch
class CollatorForGIMLETLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        self.rich_features = kwargs.pop('rich_features')
        super().__init__(**kwargs)

    def __post_init__(self):
        if self.tf_experimental_compile:
            import tensorflow as tf
            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_call(self, examples):
        graph_batch = []
        text_batch = []
        labels_batch = []

        # === 提取 SMILES 并生成 fingerprint ===
        smiles_list = [str(example_data['graph']) for example_data in examples]  # 确保每个 SMILES 是字符串类型
        fingerprints = get_morgan_fingerprint(smiles_list)  # 返回 FloatTensor (B, 2048)
        # =====================================

        for idx, example_data in enumerate(examples):
            graph_data = example_data['graph']
            graph_batch.append(graph_data)

            text_batch.append({
                'input_ids': example_data['input_ids'],
                'attention_mask': example_data['attention_mask'],
            })

            decoder_attention_mask = example_data.get('decoder_attention_mask')

            # 若为 None，则用与 labels 同 shape 的全 1 mask（或根据 labels != -100 生成更合理）
            if decoder_attention_mask is None and example_data['labels'] is not None:
                decoder_attention_mask = (torch.tensor(example_data['labels']) != -100).long().tolist()

            labels_batch.append({
                'labels': example_data['labels'],
                'decoder_attention_mask': decoder_attention_mask
            })

        # 图数据 + 文本 Padding
        graph_batch = collator_graph_data(
            graph_batch,
            transform_in_collator=self.transform_in_collator,
            rich_features=self.rich_features,
        )
        text_batch = padding(
            text_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        labels_batch = padding(
            labels_batch, self.tokenizer.pad_token_id, self.tokenizer.pad_token_type_id,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # 构造最终 Batch
        batch = {
            'graph': graph_batch,
            'input_ids': text_batch.data['input_ids'],
            'attention_mask': text_batch.data['attention_mask'],
            'labels': labels_batch.data['labels'],
            'fingerprint': fingerprints,  # ✅ 加入生成的 fingerprint
        }
        if 'decoder_attention_mask' in labels_batch.data:
            batch['decoder_attention_mask'] = labels_batch.data['decoder_attention_mask']

        return batch


