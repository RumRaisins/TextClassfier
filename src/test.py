from importlib import import_module

import torch
from transformers import BertTokenizer

if __name__ == '__main__':
    model_name = "intent"
    #model_name = "slot"
    text = list("请给我介绍一个北京的铜锅涮肉店")
    x = import_module(model_name + "_model")
    config = x.Config()
    dataset = import_module("utils.bert_dataset_" + model_name)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    if model_name == "intent":
        train_dataset = dataset.BertDataset(config.train_path, config.label_path, tokenizer=tokenizer,
                                    need_label_weight=True)
        model = x.Model(config, train_dataset.label_weight).to(config.device)
    else:
        model = x.Model(config, config.class_label).to(config.device)



    model.load_state_dict(torch.load(config.save_path))



    text_dict = tokenizer.encode_plus(
        text,
        add_special_token=True,
        max_length=128,
        ad_to_max_length=True,
        return_attention_mask=True
    )
    input_ids, attention_mask, token_type_ids = text_dict['input_ids'], \
                                                text_dict['attention_mask'], \
                                                text_dict['token_type_ids']
    trains = torch.tensor(input_ids).to(config.device).view(1, -1)
    mask = torch.tensor(attention_mask).to(config.device).view(1, -1)
    tokens = torch.tensor(token_type_ids).to(config.device).view(1, -1)

    model.eval()
    logits = model((trains, mask, tokens))



    print(text)
    if model_name == "slot":
        print(dataset.recover_intent(config.index2label, logits[0])[1:-2])
    else:
        print(dataset.recover_intent(config.index2label, logits[0]))



