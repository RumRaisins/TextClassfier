import json
import os

import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel


class Config(object):

    def __init__(self):
        self.debug = False

        self.model_name = "bret_joint"

        self.train_path = "../data/joint/joint_train_data.json"
        self.dev_path = "../data/joint/joint_val_data.json"
        self.test_path = "../data/joint/joint_test_data.json"
        self.intent_path = "../data/joint/intent_vocab.json"
        self.slot_path = "../data/joint/tag_vocab.json"
        intent = json.loads(open(self.intent_path, "r", encoding="utf8").read())
        self.intent_label = list(intent)
        self.intent2index = dict([(x, i) for i, x in enumerate(intent)])
        self.index2intent = dict([(i, x) for i, x in enumerate(intent)])

        slot = json.loads(open(self.slot_path, "r", encoding="utf8").read())
        self.slot_label = list(slot)
        self.slot2index = dict([(x, i) for i, x in enumerate(slot)])
        self.index2slot = dict([(i, x) for i, x in enumerate(slot)])

        self.save_path = "../saved"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path += "/" + self.model_name + ".ckpt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.intent_num = len(self.intent_label)
        self.slot_num = len(self.slot_label)

        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.dropout = 0.1
        self.eps = 1e-8
        self.eval_batch = 1 if self.debug else 600
        self.require_improvement = 1200
        self.bert_path = r"D:\Work\bertMode\chinese_wwm_ext_pytorch"
        # self.bert_path = r"D:\Work\bertMode\albert_tiny_zh"
        self.bert_config = BertConfig.from_pretrained(
            self.bert_path + "/config.json"
        )
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.max_length = 128



class Model(nn.Module):

    def __init__(self, config, intent_weight=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(
            config.bert_path, config=config.bert_config
        )
        self.intent_classifier = nn.Linear(config.hidden_size, config.intent_num)
        nn.init.xavier_normal_(self.intent_classifier.weight)

        if intent_weight == None:
            self.intent_weight = torch.tensor([1.0] * config.num_class)
        self.label_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=intent_weight)

        self.slot_num = config.slot_num
        self.slot_classifier = nn.Linear(config.hidden_size, config.slot_num)
        self.slot_loss_fct = torch.nn.CrossEntropyLoss()


    def forward(self, task, intent_tensor=None, slot_tensor=None):
        content = task[0]
        mask = task[1]
        token_type_ids = task[2]
        sequence_out, pooled = self.bert(content,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        intent_logits = self.intent_classifier(pooled)
        slot_logits = self.slot_classifier(sequence_out)

        if intent_tensor is None and slot_tensor is None:
            return intent_logits, slot_logits
        intent_loss = self.label_loss_fct(intent_logits, intent_tensor)

        mask = mask.view(-1) == 1
        slot_mask = mask.view(-1) == 1
        slot_logits_view = slot_logits.view(-1, self.slot_num)[slot_mask]
        slot_tensor_view = slot_tensor.view(-1)[slot_mask]
        slot_loss = self.slot_loss_fct(slot_logits_view, slot_tensor_view)


        return intent_logits, slot_logits, intent_loss, slot_loss
