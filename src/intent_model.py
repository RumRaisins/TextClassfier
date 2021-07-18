import json
import os

import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel


class Config(object):

    def __init__(self):
        self.model_name = "bret_nlu"

        self.train_path = "../data/joint/joint_train_data.json"
        self.dev_path = "../data/joint/joint_val_data.json"
        self.test_path = "../data/joint/joint_test_data.json"
        self.label_path = "../data/joint/intent_vocab.json"
        label = json.loads(open(self.label_path, "r", encoding="utf8").read())
        self.class_label = list(label)
        self.label2index = dict([(x, i) for i, x in enumerate(label)])
        self.index2label = dict([(i, x) for i, x in enumerate(label)])

        self.save_path = "../saved"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path += "/" + self.model_name + ".ckpt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_class = len(self.class_label)

        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.dropout = 0.1
        self.eps = 1e-8
        self.require_improvement = 3
        self.bert_path = r"D:\Work\bertMode\chinese_wwm_ext_pytorch"
        # self.bert_path = r"D:\Work\bertMode\albert_tiny_zh"
        self.bert_config = BertConfig.from_pretrained(
            self.bert_path + "/config.json"
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)



class Model(nn.Module):

    def __init__(self, config, label_weight=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(
            config.bert_path, config=config.bert_config
        )
        self.fc = nn.Linear(config.bert_config.hidden_size, config.num_class)
        # init intent weight
        self.label_weight = label_weight
        if label_weight == None:
            self.label_loss_fct = torch.nn.BCEWithLogitsLoss()
            return
        self.label_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.label_weight)


    def forward(self, task, label_tensor=None):
        content = task[0]
        mask = task[1]
        token_type_ids = task[2]
        tmp, pooled = self.bert(content,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        label_logits = self.fc(pooled)
        if label_tensor is None:
            return label_logits
        loss = self.label_loss_fct(label_logits, label_tensor)
        return label_logits, loss
