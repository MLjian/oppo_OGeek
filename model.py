# -*- coding: utf-8 -*-
"""
@brief :
@interfaces :
@author : Jian
"""
from typing import Dict
import torch
import pdb
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

class OppoLWZ(Model):
    def __init__(self, vocab, similar_unit, tag_feedforward, classifier_feedforward):
        super(OppoLWZ, self).__init__(vocab)
        self.similar_unit = similar_unit
        self.tag_feedforward = tag_feedforward
        self.classifier_feedforward = classifier_feedforward

        self.accuracy = CategoricalAccuracy()
        self.f1 = F1Measure(1)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, q0, q1, q2, title, prob, tag, label=None):
        q_cat = torch.cat([q0['tokens'], q1['tokens'], q2['tokens']], dim = 0)
        t_cat = torch.cat([title['tokens'], title['tokens'], title['tokens']], dim = 0)
        q_cat = {'tokens': q_cat}
        t_cat = {'tokens': t_cat}
        sim_cat = self.similar_unit(q_cat, t_cat)

        sim_cat_re = torch.reshape(sim_cat, (3, -1)).transpose(0, 1)
        tag_out = self.tag_feedforward(tag)
        
        coms = torch.cat([sim_cat_re, prob, tag_out], dim=1)
        logits = self.classifier_feedforward(coms)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}
        if label is not None:
            loss = self.loss(logits, label)
            self.accuracy(logits, label)
            self.f1(logits, label)
            output_dict["loss"] = loss
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset), 'f1_score': self.f1.get_metric(reset)[2]}