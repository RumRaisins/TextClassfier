import json

import numpy as np
import torch
from torch.utils.data import Dataset




class BertDataset(Dataset):

    def __init__(self, data_path, config, need_intent_weight=False):
        super(BertDataset, self).__init__()
        self.data = json.loads(open(data_path, "r", encoding="utf8").read())
        if config.debug:
            self.data = self.data[0:100]
        self.intent2index = config.intent2index
        self.index2intent = config.index2intent
        self.slot2index = config.slot2index
        self.index2slot = config.index2slot
        self.intent_num = config.intent_num
        self.slot_num = config.slot_num
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length

        if not need_intent_weight:
            return
        self.intent_weight = [1] * len(self.intent2index)
        for data in self.data:
            for intent in data[2]:
                self.intent_weight[self.intent2index[intent]] += 1
        train_size = len(self.data)
        for intent, intent_index in self.intent2index.items():
            neg_pos = (
                train_size - self.intent_weight[intent_index]
            ) / self.intent_weight[intent_index]
            self.intent_weight[intent_index] = np.log10(neg_pos)
        self.intent_weight = torch.tensor(self.intent_weight)




    def __getitem__(self, item):
        data = self.data[item]
        text = data[0]
        intents = [0.0] * self.intent_num
        for intent in data[2]:
            intents[self.intent2index[intent]] = 1.0
        if len(data[2]) == 0:
            # print("".join(text) + " 无标签")
            intents[self.intent2index["Other"]] = 1.0

        slots = []
        index = 0
        for label in data[1]:
            slots.append(self.slot2index[label])
            index += 1
            if index == self.max_length:
                break


        text_dict = self.tokenizer.encode_plus(
            text,
            add_special_token=True,
            max_length=self.max_length,
            ad_to_max_length=True,
            return_attention_mask=True
        )
        input_ids, attention_mask, token_type_ids = text_dict['input_ids'], \
                                                    text_dict['attention_mask'], \
                                                    text_dict['token_type_ids']
        output = {
            "token_ids": input_ids,
            'attention_mask': attention_mask,
            "token_type_ids": token_type_ids,
            "intents": intents,
            "slots": slots
        }
        return output

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和intent
    """

    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])  # batch中样本的最大的长度
    intents = torch.tensor([data["intents"] for data in batch])
    slots = [data["slots"] for data in batch]
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    # 填充每个batch的sample
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)
    slots = padding(slots, max_length)
    return token_ids_padded, attention_mask_padded, token_type_ids_padded, intents, slots


def recover_intent(index2intent, intent_logits):
    das = []
    for j in range(len(index2intent)):
        if intent_logits[j] > 0:
            intent_domain = index2intent[j]
            das.append(intent_domain)
    return das



