# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
import torch
import torch.optim as optim
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from utils import OppoDatasetReader
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.modules.bimpm_matching import BiMpmMatching
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator
import torch.nn as nn
from utils import BiMpmUnit, TagFF
from model import OppoLWZ
torch.manual_seed(1)

"""
1 加载数据
"""
reader = OppoDatasetReader()
train_dataset = reader.read('data/df_train.pkl')
validation_dataset = reader.read('data/df_valid.pkl')

"""
2 构造词表
"""
vocab = Vocabulary.from_instances(train_dataset)
vocab.save_to_files('./vocab/')

"""
3 定义批迭代器
"""
iterator = BucketIterator(batch_size=128, sorting_keys=[("title", "num_tokens")])
iterator.index_with(vocab)

"""
4 构建网络
"""
"""embedding模块"""
embedding_setting = {
        "pretrained_file": "./word2vec/sgns.zhihu_pro.bigram-char",
        "embedding_dim": 300,
        "trainable": False,
        "padding_index": 0
      }
params_embedding = Params(embedding_setting)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size("tokens"), embedding_dim=300).from_params(vocab, params_embedding)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

"""相似度计算模块"""
activ_relu = nn.ReLU()
activ_linear = lambda x: x
similar_bimpm = BiMpmUnit(vocab = vocab,
              text_field_embedder = word_embeddings,
              encoder1 = PytorchSeq2SeqWrapper(nn.LSTM(input_size=300, hidden_size=100, bidirectional=True, batch_first=True)),
              matcher_forward1 = BiMpmMatching(is_forward=True, hidden_dim=100, num_perspectives=10),
              matcher_backward1 = BiMpmMatching(is_forward=False, hidden_dim=100, num_perspectives=10),
              aggregator = PytorchSeq2VecWrapper(nn.LSTM(input_size=110, hidden_size=100, bidirectional=True, num_layers=2, batch_first=True, dropout=0.5)),
              classifier_feedforward = FeedForward(input_dim=400, num_layers=2, hidden_dims=[200, 1], activations=[activ_relu, activ_linear], dropout=[0.5, 0]),
              initializer = InitializerApplicator(),
              regularizer = None)
"""tag处理模块"""
tag_ff = TagFF(vocab, word_embeddings, FeedForward(input_dim=300, num_layers=3, hidden_dims=[100, 100, 10], activations=[activ_relu, activ_relu, activ_relu]))

"""定义模型"""
model = OppoLWZ(vocab=vocab, similar_unit=similar_bimpm, tag_feedforward=tag_ff, classifier_feedforward=FeedForward(input_dim=16, num_layers=4, hidden_dims=[100, 100, 100, 2], activations=[activ_relu, activ_relu, activ_relu, activ_linear]))

"""
5 定义优化器
"""
optimizer = optim.Adam(model.parameters(), lr=0.0001)

"""
6 训练
"""
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  cuda_device = 0,
                  patience=4,
                  num_epochs=15,
                  serialization_dir = './checkpoints')
trainer.train()