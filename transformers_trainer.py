import argparse
from src.config import Config, evaluate_batch_insts
import time
from termcolor import colored
from src.model import SynDepAT
import torch
from typing import List
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
from src.config import get_metric
from src.config.config import PaserModeType, DepModelType

from tqdm import tqdm
from collections import Counter
from src.data import TransformersNERDataset
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer
from src.config.span_eval import span_f1,span_f1_prune,get_predict,get_predict_prune
from logger import get_logger
from termcolor import colored

logger = get_logger()

def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:1", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--nerdp_dataset', type=str, default="chinese")
    parser.add_argument('--ner_dataset', type=str, default="resume")
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--nerdp_batch_size', type=int, default=8, help="ontontes is 15, chinese is 8")
    parser.add_argument('--ner_batch_size', type=int, default=15, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--nerdp_max_entity_length', type=int, default=16, help="in ner+dp domain the max span length, ontonotes and chinese is 16")
    parser.add_argument('--ner_max_entity_length', type=int, default=28, help="in ner domain the max span length, conll03 is 10, ace04 is 33, ace05 is 27, resume is 28")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--max_no_incre', type=int, default=10, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")
    parser.add_argument('--fp16', type=int, choices=[0, 1], default=0, help="use 16-bit floating point precision instead of 32-bit")

    ##model hyperparameter
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--embedder_type', type=str, default="hfl/chinese-electra-180g-large-discriminator", help="you can use 'roberta-base' and so on")
    parser.add_argument('--add_iobes_constraint', type=int, default=0, choices=[0,1], help="add IOBES constraint for transition parameters to enforce valid transitions")
    parser.add_argument("--print_detail_f1", type= int, default= 0, choices= [0, 1], help= "Open and close printing f1 scores for each tag after each evaluation epoch")
    parser.add_argument("--earlystop_atr", type=str, default="micro", choices= ["micro", "macro"], help= "Choose between macro f1 score and micro f1 score for early stopping evaluation")
    parser.add_argument('--dep_model', type=str, default="dggcn", choices=["none", "dggcn", "aelgcn"], help="dg_gcn mode consists of both GCN and Syn-LSTM")
    parser.add_argument('--nerdp_parser_mode', type=str, default="span", choices=["crf", "span"], help="parser model consists of crf and span")
    parser.add_argument('--ner_parser_mode', type=str, default="span", choices=["crf", "span"], help="parser model consists of crf and span")

    parser.add_argument("--shared_enc_type", default='adatrans', choices=['transformer', 'adatrans'])
    parser.add_argument("--attn_layer", type=int, default=6, help='the number of shared encoder layer')
    parser.add_argument('--fusion_type', type=str, default='gate-concat', choices=['concat', 'add', 'concat-add', 'gate-add', 'gate-concat'])
    parser.add_argument('--fusion_dropout', type=float, default=0.2)
    parser.add_argument('--advloss_dropout', type=float, default=0.7)


    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, nerdp_train_loader: DataLoader, nerdp_dev_loader: DataLoader, nerdp_test_loader: DataLoader,
                ner_train_loader: DataLoader, ner_dev_loader: DataLoader, ner_test_loader: DataLoader):
    ### Data Processing Info
    if len(nerdp_train_loader) >= len(ner_train_loader):
        train_num = len(nerdp_train_loader)
    else:
        train_num = len(ner_train_loader)

    print(colored(f"[Info] number of training nerdp instances: {len(nerdp_train_loader)}", "yellow"))
    print(colored(f"[Info] number of training ner instances: {len(ner_train_loader)}", "red"))
    print(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}")
    model = SynDepAT(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(model=model, learning_rate=config.learning_rate,
                                                                   num_training_steps=train_num * epoch,
                                                                   weight_decay=0.0, eps = 1e-8, warmup_step=0)
    print(f"[Optimizer Info] Modify the optimizer info as you need.")
    print(optimizer)

    model.to(config.device)
    scaler = None
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))

    best_ner_dev = [-1, 0]
    best_ner_test = [-1, 0]
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        model.train()
        # ner + dp
        for iter, batch in tqdm(enumerate(nerdp_train_loader, 1), total=len(nerdp_train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                if config.nerdp_parser_mode == PaserModeType.span:
                    loss = model(Task = True, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
                                 all_span_lens=batch.all_span_lens.to(config.device), all_span_ids = batch.all_span_ids.to(config.device),
                                 all_span_weight=batch.all_span_weight.to(config.device), real_span_mask=batch.all_span_mask.to(config.device),
                                 labels = batch.label_ids.to(config.device))
                else:
                    loss = model(Task = True, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
                                 all_span_lens=None, all_span_ids=None, all_span_weight=None, real_span_mask=None,
                                 labels = batch.label_ids.to(config.device))
            epoch_loss += loss.item()
        if config.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if config.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # only ner
        for iter, batch in tqdm(enumerate(ner_train_loader, 1), total=len(ner_train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                if config.ner_parser_mode == PaserModeType.span:
                    loss = model(Task = False, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=None, deplabels=None,
                                 all_span_lens=batch.all_span_lens.to(config.device), all_span_ids = batch.all_span_ids.to(config.device),
                                 all_span_weight=batch.all_span_weight.to(config.device), real_span_mask=batch.all_span_mask.to(config.device),
                                 labels = batch.label_ids.to(config.device))
                else:
                    loss = model(Task = False, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=None, deplabels=None,
                                 all_span_lens=None, all_span_ids=None, all_span_weight=None, real_span_mask=None,
                                 labels = batch.label_ids.to(config.device))
            epoch_loss += loss.item()
            if config.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if config.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            model.zero_grad()
        end_time = time.time()
        logger.info(f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        ner_dev_metrics = ner_evaluate_model(config, model, ner_dev_loader, "dev", ner_dev_loader.dataset.insts)
        ner_test_metrics = ner_evaluate_model(config, model, ner_test_loader, "test", ner_test_loader.dataset.insts)

        if ner_dev_metrics[2] > best_ner_dev[0]:
            no_incre_dev = 0
            best_ner_dev[0] = ner_dev_metrics[2]
            best_ner_dev[1] = i
            best_ner_test[0] = ner_test_metrics[2]
            best_ner_test[1] = i
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            logger.info("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

def ner_evaluate_model(config: Config, model: SynDepAT, data_loader: DataLoader, name: str, insts: List, print_each_type_metric: bool = False):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    total_correct, total_predict, total_golden = 0, 0, 0
    batch_size = data_loader.batch_size
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config.fp16)):
        for batch_id, batch in enumerate(data_loader, 0):
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            if config.ner_parser_mode == PaserModeType.span:
                logits = model(Task = False, subword_input_ids=batch.input_ids.to(config.device),
                             word_seq_lens=batch.word_seq_len.to(config.device),
                             orig_to_tok_index=batch.orig_to_tok_index.to(config.device), attention_mask=batch.attention_mask.to(config.device),
                             depheads=None, deplabels=None,
                             all_span_lens=batch.all_span_lens.to(config.device), all_span_ids=batch.all_span_ids.to(config.device),
                             all_span_weight=batch.all_span_weight.to(config.device), real_span_mask=batch.all_span_mask.to(config.device),
                             labels=batch.label_ids.to(config.device), is_train=False)
                batch_all_real_span_ids = []
                for i in range(batch.all_span_ids.size(0)):
                    selected_ids = batch.all_span_ids[i][batch.all_span_mask[i].nonzero(as_tuple=True)]
                    selected_ids_tuple = [tuple(map(int, coord)) for coord in selected_ids.tolist()]
                    batch_all_real_span_ids.append(selected_ids_tuple)
                span_f1s = span_f1_prune(batch_all_real_span_ids, logits,
                                                         batch.label_ids.to(config.device), batch.all_span_mask.to(config.device))
                batch_correct, batch_pred, batch_golden = span_f1s
                total_correct += batch_correct.item()
                total_predict += batch_pred.item()
                total_golden += batch_golden.item()
                batch_id += 1
            else:
                logits = model(Task = False, subword_input_ids=batch.input_ids.to(config.device),
                             word_seq_lens=batch.word_seq_len.to(config.device),
                             orig_to_tok_index=batch.orig_to_tok_index.to(config.device),
                             attention_mask=batch.attention_mask.to(config.device),
                             depheads=None, deplabels=None,
                             all_span_lens=None, all_span_ids=None, all_span_weight=None, real_span_mask=None,
                             labels=batch.label_ids.to(config.device), is_train=False)
                batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, logits, batch.label_ids, batch.word_seq_len, config.ner_idx2labels)
                p_dict += batch_p
                total_predict_dict += batch_predict
                total_entity_dict += batch_total
                batch_id += 1
    if config.ner_parser_mode == PaserModeType.crf:
        f1Scores = []
        if print_each_type_metric or config.print_detail_f1 or (config.earlystop_atr == "macro"):
            for key in total_entity_dict:
                precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
                logger.info(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")
                f1Scores.append(fscore_key)
            if len(f1Scores) > 0:
                logger.info(f"[{name} set Total] Macro F1: {sum(f1Scores) / len(f1Scores):.2f}")

        total_p = sum(list(p_dict.values()))
        total_predict = sum(list(total_predict_dict.values()))
        total_entity = sum(list(total_entity_dict.values()))
        precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
        logger.info(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}")

        if config.earlystop_atr == "macro" and len(f1Scores) > 0:
            fscore = sum(f1Scores) / len(f1Scores)
    else: # PaserModeType.span
        precision =total_correct / (total_predict+1e-10) * 100
        recall = total_correct / (total_golden + 1e-10) * 100
        fscore = precision * recall * 2 / (precision + recall + 1e-10)
        logger.info(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}")
    return [precision, recall, fscore]

def main():
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    # NER+DP Data Reading
    tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True)
    logger.info(f"[NER+DP Data Info] Reading dataset from: \t{conf.nerdp_train_file}\t{conf.nerdp_dev_file}\t{conf.nerdp_test_file}")
    nerdp_train_dataset = TransformersNERDataset(conf.nerdp_max_entity_length, conf.nerdp_parser_mode, conf.dep_model, conf.nerdp_train_file, tokenizer,
                                           number=-1, is_train=True)
    conf.nerdp_label2idx = nerdp_train_dataset.label2idx
    conf.nerdp_idx2labels = nerdp_train_dataset.idx2labels
    conf.deplabel2idx = nerdp_train_dataset.deplabel2idx
    conf.root_dep_label_id = nerdp_train_dataset.root_dep_label_id
    # conf.nerdp_poslabel2idx = nerdp_train_dataset.poslabel2idx
    conf.nerdp_label_size = len(nerdp_train_dataset.label2idx)

    nerdp_dev_dataset = TransformersNERDataset(conf.nerdp_max_entity_length, conf.nerdp_parser_mode, conf.dep_model, conf.nerdp_dev_file, tokenizer,
                                         number=-1, label2idx=nerdp_train_dataset.label2idx,
                                         deplabel2idx=nerdp_train_dataset.deplabel2idx, is_train=False)
    nerdp_test_dataset = TransformersNERDataset(conf.nerdp_max_entity_length, conf.nerdp_parser_mode, conf.dep_model, conf.nerdp_test_file, tokenizer,
                                          number=-1, label2idx=nerdp_train_dataset.label2idx,
                                          deplabel2idx=nerdp_train_dataset.deplabel2idx, is_train=False)
    num_workers = 0
    # conf.nerdp_max_entity_length = max(max(nerdp_train_dataset.max_entity_length, nerdp_dev_dataset.max_entity_length),
    #                              nerdp_test_dataset.max_entity_length)
    nerdp_train_dataloader = DataLoader(nerdp_train_dataset, batch_size=conf.nerdp_batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=nerdp_train_dataset.collate_to_max_length)
    nerdp_dev_dataloader = DataLoader(nerdp_dev_dataset, batch_size=conf.nerdp_batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=nerdp_dev_dataset.collate_to_max_length)
    nerdp_test_dataloader = DataLoader(nerdp_test_dataset, batch_size=conf.nerdp_batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=nerdp_test_dataset.collate_to_max_length)

    # Only NER Data Reading
    # tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True, use_fast=True)
    logger.info(f"[Only NER Data Info] Reading dataset from: \t{conf.ner_train_file}\t{conf.ner_dev_file}\t{conf.ner_test_file}")
    ner_train_dataset = TransformersNERDataset(conf.ner_max_entity_length, conf.ner_parser_mode, DepModelType.none, conf.ner_train_file, tokenizer, number=-1, is_train=True)
    conf.ner_label2idx = ner_train_dataset.label2idx
    conf.ner_idx2labels = ner_train_dataset.idx2labels
    conf.ner_label_size = len(ner_train_dataset.label2idx)
    ner_dev_dataset = TransformersNERDataset(conf.ner_max_entity_length, conf.ner_parser_mode, DepModelType.none, conf.ner_dev_file, tokenizer,
                                         number=-1, label2idx=ner_train_dataset.label2idx, deplabel2idx=None, is_train=False)

    ner_test_dataset = TransformersNERDataset(conf.ner_max_entity_length, conf.ner_parser_mode, DepModelType.none, conf.ner_test_file, tokenizer,
                                         number=-1, label2idx=ner_train_dataset.label2idx, deplabel2idx=None, is_train=False)

    # conf.ner_max_entity_length = max(max(ner_train_dataset.max_entity_length,  ner_dev_dataset.max_entity_length),
    #                              ner_test_dataset.max_entity_length)

    ner_train_dataloader = DataLoader(ner_train_dataset, batch_size=conf.ner_batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=ner_train_dataset.collate_to_max_length)
    ner_dev_dataloader = DataLoader(ner_dev_dataset, batch_size=conf.ner_batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=ner_dev_dataset.collate_to_max_length)
    ner_test_dataloader = DataLoader(ner_test_dataset, batch_size=conf.ner_batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=ner_test_dataset.collate_to_max_length)
    # if conf.nerdp_parser_mode == PaserModeType.span:
    #     print(colored(f"[INFO] nerdp_max_entity_length: {conf.nerdp_max_entity_length}", "green"))
    # if conf.ner_parser_mode == PaserModeType.span:
    #     print(colored(f"[INFO] ner_max_entity_length: {conf.ner_max_entity_length}", "magenta"))
    train_model(conf, conf.num_epochs, nerdp_train_dataloader, nerdp_dev_dataloader, nerdp_test_dataloader,
                ner_train_dataloader, ner_dev_dataloader, ner_test_dataloader)

if __name__ == "__main__":
    main()
