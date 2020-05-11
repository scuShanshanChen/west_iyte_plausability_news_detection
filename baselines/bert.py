from transformers import BertTokenizer, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, TensorDataset, random_split
import math
import numpy as np
import pandas as pd
import os
from joblib import Memory
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from utils.dl_runner import save_model

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'

logger = logging.getLogger('baselines/bert.py')




def read_files(args):
    train_path = os.path.join(args.target_path, 'train.tsv')
    dev_path = os.path.join(args.target_path, 'dev.tsv')
    test_path = os.path.join(args.target_path, 'test.tsv')


    logging.debug("Reading Datasets...")
    df_train = pd.read_csv(train_path, delimiter='\t')
    df_dev = pd.read_csv(dev_path, delimiter='\t')
    df_test = pd.read_csv(test_path, delimiter='\t')


    train_articles = df_train.text.values
    train_labels = df_train.label.values
    dev_articles = df_dev.text.values
    dev_labels = df_dev.label.values
    test_articles = df_test.text.values
    test_labels = df_test.label.values

    logging.debug("Tokenizing the Dataset for BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_ids = []
    train_att_mask = []
    dev_ids = []
    dev_att_mask = []
    test_ids = []
    test_att_mask = []
    for article in train_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True,max_length=args.MAX_LEN, pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
        train_ids.append(encoded_article['input_ids'])
        train_att_mask.append(encoded_article['attention_mask'])
    for article in dev_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True, return_tensors='pt')
        dev_ids.append(encoded_article['input_ids'])
        dev_att_mask.append(encoded_article['attention_mask'])
    for article in test_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True,max_length=args.MAX_LEN, pad_to_max_length=True,
                                                return_attention_mask=True, return_tensors='pt')
        test_ids.append(encoded_article['input_ids'])
        test_att_mask.append(encoded_article['attention_mask'])

    train_ids = torch.cat(train_ids, dim=0)
    dev_ids = torch.cat(dev_ids, dim=0)
    test_ids = torch.cat(test_ids, dim=0)
    train_att_mask = torch.cat(train_att_mask, dim=0)
    dev_att_mask = torch.cat(dev_att_mask, dim=0)
    test_att_mask = torch.cat(test_att_mask, dim=0)
    train_labels = torch.tensor(train_labels)
    dev_labels = torch.tensor(dev_labels)
    test_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_ids,train_att_mask,train_labels)
    dev_dataset = TensorDataset(dev_ids,dev_att_mask,dev_labels)
    test_dataset = TensorDataset(test_ids,test_att_mask,test_labels)

    return train_dataset, dev_dataset, test_dataset





def train_split(model, train_data, dev_data, optimizer, args):

    train_iter = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
    dev_iter = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.batch_size)

    torch.cuda.empty_cache()

    logging.info(
        "Number of training samples {train}, number of dev samples {dev} in random split".format(
            train=len(train_iter),
            dev=len(dev_iter)))

    train(train_iter, dev_iter, model, optimizer, args)







def train(train_iter, dev_iter, model, optimizer, args):
    best_dev_f1 = -1

    n_total_steps = len(train_iter)
    total_iter = len(train_iter)*args.epochs

    logging.info(
        stats_columns.format(
            'Epoch', 'T-Acc', 'T-F1', 'T-Recall', 'T-Prec', 'T-Loss'
            , 'D-Acc', 'D-F1', 'D-Recall', 'D-Prec', 'D-Loss'))


    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

    for epoch in range(args.epochs):

        model.train()

        train_loss = 0
        preds = []
        trues = []

        for batch_ids in train_iter:

            input_ids = batch_ids[0].to(args.device)
            att_masks = batch_ids[1].to(args.device)
            labels = batch_ids[2].to(args.device)
            labels.to(args.device)

            model.zero_grad()

            # forward pass
            loss, logits =  model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

            # record preds, trues
            _pred = logits.cpu().data.numpy()
            preds.append(_pred)
            _label = labels.cpu().data.numpy()
            trues.append(_label)

            train_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec = calculate_metrics(trues, preds)


        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, args)

        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred)

        logging.info(
            stats_columns.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss, dev_acc, dev_f1,
                                 dev_recall, dev_prec, dev_loss))

        if best_dev_f1 < dev_f1:
            logging.debug('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(dev_acc=dev_f1,
                                                                                                    best_dev_acc=best_dev_f1))
            best_dev_f1 = dev_f1

            training_mode = args.training_mode
            model_name = '{training_mode}_epoch_{epoch}_dev_f1_{dev_f1:03}.pth.tar'.format(training_mode=training_mode,
                                                                                           epoch=epoch,
                                                                                           dev_f1=dev_f1)
            save_model(model, optimizer, epoch, model_name, training_mode, args.checkpoint_dir)



def eval(dev_iter, model, args):
    n_total_steps = len(dev_iter)
    model.eval()
    dev_loss = 0
    preds = []
    trues = []
    for batch_ids in dev_iter:
        input_ids = batch_ids[0].to(args.device)
        att_masks = batch_ids[1].to(args.device)
        labels = batch_ids[2].to(args.device)
        labels.to(args.device)

        # forward pass
        with torch.no_grad():
            loss, logits =  model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    dev_loss = dev_loss / n_total_steps
    return trues, preds, dev_loss





def calculate_metrics(label, pred):
    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
    label_class = np.concatenate([numarray for numarray in label]).ravel()

    logging.debug('Expected: \n{}'.format(label_class[:20]))
    logging.debug('Predicted: \n{}'.format(pred_class[:20]))
    acc = accuracy_score(label_class, pred_class)
    f1 = f1_score(label_class, pred_class, average='binary')
    recall = recall_score(label_class, pred_class)
    prec = precision_score(label_class, pred_class)

    return acc, f1, recall, prec







def add_bert_specific_parser(parser):
    parser.add_argument('--MAX_LEN', type=int, default=512)
    return parser
