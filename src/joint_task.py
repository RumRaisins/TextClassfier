from datetime import time

import numpy as np
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW

import src.joint_model as model
from src.utils.bert_dataset_joint import BertDataset, collate_fn, recover_intent
from src.utils.bert_dataset_slot import recover_slot
from src.utils.tools import create_logger, calculate_f1


def train(config, model, train_iter, dev_iter, test_iter):
    write = SummaryWriter("../runs")

    model.train()
    print('User AdamW...')
    print(config.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      eps=config.eps)
    total_batch = 0  # 记录进行到多少batch
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    dev_best_F1 = float('inf')
    for epoch in range(config.num_epochs):
        print('\nEpoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        number_batchs = len(train_iter)
        with tqdm(total=number_batchs // config.batch_size) as batch_progress:
            for i, (trains, mask, tokens, intents, slots) in enumerate(tqdm(train_iter)):
                trains = trains.to(config.device)
                intents = intents.to(config.device)
                slots = slots.to(config.device)
                mask = mask.to(config.device)
                tokens = tokens.to(config.device)

                model.zero_grad()
                _,_, intent_loss, slot_loss = model((trains, mask, tokens), intents, slots)
                loss = intent_loss + slot_loss
                loss.backward()
                optimizer.step()
                write.add_scalar(config.model_name + " intent", intent_loss.item(), global_step=total_batch)
                write.add_scalar(config.model_name + " slot", slot_loss.item(), global_step=total_batch)
                write.add_scalar(config.model_name + " total", loss.item(), global_step=total_batch)
                if total_batch % 100 == 0:
                    batch_progress.set_description(f'Epoch {epoch}')
                    batch_progress.set_postfix(Batch=total_batch,
                                               loss=loss.item())
                    batch_progress.update()
                if total_batch % config.eval_batch == 0 and total_batch != 0:
                    F1 = evaluate(config, model, dev_iter)
                    if F1 < dev_best_F1:
                        dev_best_F1 = F1
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                        print("best val F1 %.4f" % dev_best_F1)
                        print("save on", config.save_path)
                    # else:
                    #     print("")
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
        if flag:
            break
    evaluate(config, model, test_iter, test=True)
    return 0

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for texts, mask, tokens, intent_tensor, slot_tensor in tqdm(data_iter):
            texts = texts.to(config.device)
            intent_tensor = intent_tensor.to(config.device)
            slot_tensor = slot_tensor.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            intent_logits, slot_logits, intent_loss, slot_loss = model((texts, mask, tokens), intent_tensor, slot_tensor)
            loss_total += (intent_loss + slot_loss)
            result_intent = []
            result_ner = []
            for i in range(intent_logits.shape[0]):
                predicts = recover_intent(config.index2intent, intent_logits[i])
                origin_label = recover_intent(config.index2intent, intent_tensor[i])
                result_intent.append({
                    "predict": predicts,
                    "label": origin_label
                })

                predicts = recover_slot(config.index2slot, slot_logits[i])
                origin_label = recover_slot(config.index2slot, slot_tensor[i])
                result_ner.append({
                    "predict": predicts,
                    "label": origin_label
                })

        total = len(data_iter)
        print("%d samples val" % total)
        print("\t val loss:", loss_total/total)
        if test:
            print("!" * 20 + "test" + "!" * 20)
        precision, recall, F1_intent = calculate_f1(result_intent)
        print("-" * 20 + "intent" + "-" * 20)
        print("\t Precision: %.2f" % (100 * precision))
        print("\t Recall: %.2f" % (100 * recall))
        print("\t F1: %.2f" % (100 * F1_intent))

        precision, recall, F1_slot = calculate_f1(result_ner)
        print("-" * 20 + "slot" + "-" * 20)
        print("\t Precision: %.2f" % (100 * precision))
        print("\t Recall: %.2f" % (100 * recall))
        print("\t F1: %.2f" % (100 * F1_slot))



        return F1_intent + F1_slot


if __name__ == '__main__':
    config = model.Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    logger = create_logger('../logs/train.log')

    logger.info('Building tokenizer')
    print('config.bert_path is ', config.bert_path)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    logger.info('Loading dataset')
    train_dataset = BertDataset(config.train_path, config, need_intent_weight=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)
    dev_dataset = BertDataset(config.dev_path, config)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True)
    test_dataset = BertDataset(config.test_path, config)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn)

    logger.info('load network')
    model = model.Model(config, train_dataset.intent_weight).to(config.device)

    logger.info('training model')
    train(config, model, train_dataloader, dev_dataloader, test_dataloader)
