import json
import os

import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel


class Config(object):

    def __init__(self):
        self.model_name = "bret_slot"

        self.train_path = "../data/joint_train_data.json"
        self.dev_path = "../data/joint_val_data.json"
        self.test_path = "../data/joint_test_data.json"
        self.label_path = "../data/tag_vocab.json"
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
        self.bert_config = BertConfig.from_pretrained(
            self.bert_path + "/config.json"
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768



class Model(nn.Module):

    def __init__(self, config, slot_num):
        super(Model, self).__init__()
        self.slot_num = slot_num
        self.bert = BertModel.from_pretrained(
            config.bert_path, config=config.bert_config
        )
        self.fc = nn.Linear(config.hidden_size, config.num_class)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, task, slot_tensor=None):
        content = task[0]
        mask = task[1]
        token_type_ids = task[2]
        sequence_output, pooled = self.bert(content,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, slot_num]
        slot_logits = self.fc(sequence_output)

        if slot_tensor is None:
            return slot_logits

        # slot_tensor [batch_size, seq_length, slot_num]
        slot_logits_view = slot_logits.view(-1, self.slot_num)
        slot_tensor_view = slot_tensor.view(-1)
        loss = self.slot_loss_fct(slot_logits_view, slot_tensor_view)
        return slot_logits, loss
