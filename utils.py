# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import numpy as np
import torch
import pickle
import pdb
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, LabelField, ArrayField
from typing import Dict, Optional, List
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.modules.bimpm_matching import BiMpmMatching
from typing import Iterable, Iterator, Callable
from allennlp.data.instance import Instance
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError

class _LazyInstances(Iterable):
    def __init__(self, instance_generator: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.instance_generator = instance_generator
    def __iter__(self) -> Iterator[Instance]:
        instances = self.instance_generator()
        if isinstance(instances, list):
            raise ConfigurationError("For a lazy dataset reader, _read() must return a generator")
        return instances

class OppoDatasetReader(DatasetReader):
    """数据读取器"""
    def __init__(self,
                 lazy: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'rb') as f:
            df_data = pickle.load(f)
            df_data = df_data.reset_index(drop=True).iloc[:1000, :]

        lazy = getattr(self, 'lazy', None)

        if lazy:
            return _LazyInstances(lambda: iter(self._read(df_data)))
        else:
            instances = self._read(df_data)
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not instances:
                raise ConfigurationError("No instances were read from the given filepath {}. "
                                         "Is the path correct?".format(file_path))
            return instances


    def _read(self, df_data):

        for i in range(len(df_data)):
            q0 = df_data.loc[i, 'q0']
            q1 = df_data.loc[i, 'q1']
            q2 = df_data.loc[i, 'q2']
           
            prob = df_data.loc[i, 'prob'][:3]
            title = df_data.loc[i, 'title']
            tag = df_data.loc[i, 'tag']
            label = df_data.loc[i, 'label']

            tokenized_q0 = list(map(Token, q0))#[Token(word) for word in q0]
            tokenized_q1 = list(map(Token, q1))#[Token(word) for word in q1]
            tokenized_q2 = list(map(Token, q2))#[Token(word) for word in q2]

            tokenized_title = list(map(Token, title))#[Token(word) for word in title]
            tokenized_tag = [Token(tag)]
            yield self.text_to_instance(tokenized_q0, tokenized_q1, tokenized_q2, prob, tokenized_title, tokenized_tag, label)
        del df_data

    def text_to_instance(self,
                         tokenized_q0: List[Token],
                         tokenized_q1: List[Token],
                         tokenized_q2: List[Token],

                         prob: List[float],
                         tokenized_title: List[Token],
                         tokenized_tag: List[Token],
                         label_y: int = None) -> Instance:

        title_field = TextField(tokenized_title, self._token_indexers)
        tag_field = TextField(tokenized_tag, self._token_indexers)
        q0_field = TextField(tokenized_q0, self._token_indexers)
        q1_field = TextField(tokenized_q1, self._token_indexers)
        q2_field = TextField(tokenized_q2, self._token_indexers)

        prob_field = ArrayField(np.array(prob))
        fields = {'title': title_field, 'q0': q0_field, 'q1': q1_field, 'q2': q2_field, 'tag': tag_field,'prob': prob_field}
        if label_y is not None:
            fields['label'] = LabelField(str(label_y))
        return Instance(fields)
        
class BiMpmUnit(Model):
    """相似度计算:bimpm"""
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder1: Seq2SeqEncoder,
                 matcher_forward1: BiMpmMatching,
                 matcher_backward1: BiMpmMatching,

                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 dropout: float = 0.1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMpmUnit, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder

        self.encoder1 = encoder1
        self.matcher_forward1 = matcher_forward1
        self.matcher_backward1 = matcher_backward1

        self.aggregator = aggregator

        matching_dim = self.matcher_forward1.get_output_dim() + self.matcher_backward1.get_output_dim()

        check_dimensions_match(matching_dim, self.aggregator.get_input_dim(),
                               "sum of dim of all matching layers", "aggregator input dim")

        self.classifier_feedforward = classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)
        initializer(self)

    def forward(self,
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor]
               ):

        mask_premise = util.get_text_field_mask(premise)
        mask_hypothesis = util.get_text_field_mask(hypothesis)

        # embedding and encoding of the premise
        embedded_premise = self.dropout(self.text_field_embedder(premise))
        encoded_premise1 = self.dropout(self.encoder1(embedded_premise, mask_premise))

        # embedding and encoding of the hypothesis
        embedded_hypothesis = self.dropout(self.text_field_embedder(hypothesis))
        encoded_hypothesis1 = self.dropout(self.encoder1(embedded_hypothesis, mask_hypothesis))

        matching_vector_premise: List[torch.Tensor] = []
        matching_vector_hypothesis: List[torch.Tensor] = []

        def add_matching_result(matcher, encoded_premise, encoded_hypothesis):
            # utility function to get matching result and add to the result list
            matching_result = matcher(encoded_premise, mask_premise, encoded_hypothesis, mask_hypothesis)

            matching_vector_premise.extend(matching_result[0])
            matching_vector_hypothesis.extend(matching_result[1])

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        half_hidden_size_1 = self.encoder1.get_output_dim() // 2

        add_matching_result(self.matcher_forward1,
                            encoded_premise1[:, :, :half_hidden_size_1],
                            encoded_hypothesis1[:, :, :half_hidden_size_1])
        add_matching_result(self.matcher_backward1,
                            encoded_premise1[:, :, half_hidden_size_1:],
                            encoded_hypothesis1[:, :, half_hidden_size_1:])
        # concat the matching vectors
        matching_vector_cat_premise = self.dropout(torch.cat(matching_vector_premise, dim=2))
        matching_vector_cat_hypothesis = self.dropout(torch.cat(matching_vector_hypothesis, dim=2))

        # aggregate the matching vectors
        aggregated_premise = self.dropout(self.aggregator(matching_vector_cat_premise, mask_premise))
        aggregated_hypothesis = self.dropout(self.aggregator(matching_vector_cat_hypothesis, mask_hypothesis))

        logits = self.classifier_feedforward(torch.cat([aggregated_premise, aggregated_hypothesis], dim=-1))
        prob = torch.nn.functional.sigmoid(logits)

        return prob

class TagFF(Model):
    """tag处理模块"""
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 tag_feedforward: FeedForward) -> None:
        super(TagFF, self).__init__(vocab)
        self.embedding = text_field_embedder
        self.ff = tag_feedforward

    def forward(self, tag):
        emb_tag = self.embedding(tag)
        out_ff = self.ff(emb_tag).squeeze(dim=1)
        #pdb.set_trace()
        return out_ff


import datetime
from contextlib import contextmanager
import time

def getFormatDate(n):
    return (datetime.datetime.now() + datetime.timedelta(days=n)).strftime('%Y%m%d')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f}s".format(title, time.time() - t0))

def getPRScore(f1_score, pos_num):
    gt_num = 74706
    tp_num = 0.5 * f1_score * (gt_num + pos_num)
    precision = tp_num / pos_num
    recall = tp_num / gt_num
    return tp_num, precision, recall
