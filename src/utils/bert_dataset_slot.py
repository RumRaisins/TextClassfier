import json

import numpy as np
import torch
from torch.utils.data import Dataset




class BertDataset(Dataset):

    def __init__(self, data_path, label_path, debug=False, tokenizer=None, max_length=128, need_label_weight=False):
        super(BertDataset, self).__init__()
        self.data = json.loads(open(data_path, "r", encoding="utf8").read())
        if debug:
            self.data = self.data[0:10000]
        label = json.loads(open(label_path, "r", encoding="utf8").read())
        self.label2index = dict([(x, i) for i, x in enumerate(label)])
        self.index2label = dict([(i, x) for i, x in enumerate(label)])
        self.label_num = len(label)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        data = self.data[item]
        text = data[0]

        slots = []
        for label in data[1]:
            slot_mark = [0.0] * self.label_num
            slot_mark[self.label2index[label]] = 1.0
            slots.append(slot_mark)

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
            "labels": slots
        }
        return output

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """

    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    def padding_slot(slot, max_length, pad_idx=0):
        slot_num = len(slot[0][0])
        result = []
        #[batch_size, seq_line, slot_num]
        for line in slot:
            for item in range(max(0, max_length - len(line))):
                line.append([pad_idx] * slot_num)
        return torch.tensor(slot)




    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])  # batch中样本的最大的长度
    labels = [data["labels"] for data in batch]
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    # 填充每个batch的sample
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)
    labels = padding_slot(labels, max_length)
    return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels


def recover_intent(index2label, label_logits):
    das = []
    for j in range(len(index2label)):
        if label_logits[j] > 0:
            label_domain = index2label[j]
            das.append(label_domain)
    return das




if __name__ == '__main__':
    tmp = BertDataset(open("../../data/intent_test_data.json","r", encoding="utf8").read(),
                      debug=True)
