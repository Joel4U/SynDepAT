import argparse
from src.config import Config, from_label_id_tensor_to_label_sequence
import time
from src.model import SynDepAT
import torch
import os
# from src.config.utils import write_results
from src.config.config import PaserModeType, DepModelType
from src.config.transformers_util import get_huggingface_optimizer_and_scheduler
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
from src.data import TransformersNERDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate.utils import set_seed
import logging
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from src.data.data_utils import PAD
from logger import get_logger
import datasets
from datasets.metric import Metric
from termcolor import colored
"""
Same as transformers_trainer.py but with distributed training.
"""

from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)

logger = get_logger()


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--dataset', type=str, default="conll03")
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=30, help="batch_size per device. For distributed training, this is the batch_size per gpu")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 100.")
    parser.add_argument('--max_no_incre', type=int, default=80, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")
    parser.add_argument('--fp16', type=int, choices=[0, 1], default=1, help="use 16-bit floating point precision instead of 32-bit")

    ##model hyperparameter
    parser.add_argument('--lstm_outputsize', type=int, default=0, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")

    parser.add_argument('--embedder_type', type=str, default="roberta-base", help="you can use 'bert-base-uncased' and so on")
    parser.add_argument('--add_iobes_constraint', type=int, default=0, choices=[0,1], help="add IOBES constraint for transition parameters to enforce valid transitions")

    parser.add_argument("--print_detail_f1", type= int, default= 0, choices= [0, 1], help= "Open and close printing f1 scores for each tag after each evaluation epoch")
    parser.add_argument("--earlystop_atr", type=str, default="micro", choices= ["micro", "macro"], help= "Choose between macro f1 score and micro f1 score for early stopping evaluation")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"], help="training model or test mode")
    parser.add_argument('--dep_model', type=str, default="dggcn", choices=["none", "dggcn", "aelgcn"], help="dg_gcn mode consists of both GCN and Syn-LSTM")
    parser.add_argument('--parser_mode', type=str, default="crf", choices=["crf", "span"], help="parser model consists of crf and span")

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

    model, optimizer, train_loader, dev_loader, test_loader, scheduler = accelerator.prepare(model, optimizer,
                                                                                             nerdp_train_loader, nerdp_dev_loader, nerdp_test_loader,
                                                                                             ner_train_loader, ner_dev_loader, ner_test_loader,scheduler)

    metric = datasets.load_metric('seqeval')
    best_ner_dev = [-1, 0]
    best_ner_test = [-1, 0]

    # model_folder = config.model_folder
    # res_folder = "results"
    # if os.path.exists("model_files/" + model_folder):
    #     raise FileExistsError(
    #         f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
    #         f"to avoid override.")
    # model_path = f"model_files/{model_folder}/lstm_crf.m"
    # config_path = f"model_files/{model_folder}/config.conf"
    # res_path = f"{res_folder}/{model_folder}.results"
    # logger.info("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    # os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    # os.makedirs(res_folder, exist_ok=True)
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
                if config.parser_mode == PaserModeType.span:
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
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
        # only ner
        for iter, batch in tqdm(enumerate(ner_train_loader, 1), total=len(ner_train_loader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                if config.parser_mode == PaserModeType.span:
                    loss = model(Task = False, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
                                 all_span_lens=batch.all_span_lens.to(config.device), all_span_ids = batch.all_span_ids.to(config.device),
                                 all_span_weight=batch.all_span_weight.to(config.device), real_span_mask=batch.all_span_mask.to(config.device),
                                 labels = batch.label_ids.to(config.device))
                else:
                    loss = model(Task = False, subword_input_ids = batch.input_ids.to(config.device),
                                 word_seq_lens = batch.word_seq_len.to(config.device),
                                 orig_to_tok_index = batch.orig_to_tok_index.to(config.device),
                                 attention_mask = batch.attention_mask.to(config.device),
                                 depheads=batch.dephead_ids.to(config.device), deplabels=batch.deplabel_ids.to(config.device),
                                 all_span_lens=None, all_span_ids=None, all_span_weight=None, real_span_mask=None,
                                 labels = batch.label_ids.to(config.device))
            epoch_loss += loss.item()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()
        end_time = time.time()
        logger.info(f"Epoch {i}: {epoch_loss:.5f}, Time is {(end_time - start_time):.2f}s")

        model.eval()
        dev_metrics = ner_evaluate_model(config, model, dev_loader, "dev", metric)
        test_metrics = ner_evaluate_model(config, model, test_loader, "test", metric)
        if dev_metrics[2] > best_ner_dev[0]:
            logger.info(f"saving the best model with best dev f1 score {dev_metrics[2]}")
            no_incre_dev = 0
            best_ner_dev[0] = dev_metrics[2]
            best_ner_dev[1] = i
            best_ner_test[0] = test_metrics[2]
            best_ner_test[1] = i
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            logger.info("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break


def ner_evaluate_model(config: Config,model: SynDepAT, data_loader: DataLoader, name: str, metric:Metric, print_each_type_metric: bool = False):
    ## evaluation
    all_predictions = []
    all_golds = []
    insts = data_loader.dataset.insts
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(config.fp16)):
        for batch_id, batch in enumerate(data_loader, 0):
            if config.parser_mode == PaserModeType.crf:
                logits = model(Task = False, subword_input_ids=batch.input_ids.to(config.device),
                             word_seq_lens=batch.word_seq_len.to(config.device),
                             orig_to_tok_index=batch.orig_to_tok_index.to(config.device), attention_mask=batch.attention_mask.to(config.device),
                             depheads=None, deplabels=None,
                             all_span_lens=batch.all_span_lens.to(config.device), all_span_ids=batch.all_span_ids.to(config.device),
                             all_span_weight=batch.all_span_weight.to(config.device), real_span_mask=batch.all_span_mask.to(config.device),
                             labels=batch.label_ids.to(config.device), is_train=False)

                batch_max_ids = accelerator.pad_across_processes(logits, dim=1, pad_index=config.ner_label2idx[PAD])
                batch_max_ids = accelerator.gather_for_metrics(batch_max_ids)

                batch_label_ids = accelerator.pad_across_processes(batch.label_ids, dim=1, pad_index=config.ner_label2idx[PAD])
                batch_label_ids = accelerator.gather_for_metrics(batch_label_ids)

                word_seq_lens = accelerator.gather_for_metrics(batch.word_seq_len)
                predict_sequences = from_label_id_tensor_to_label_sequence(batch_ids = batch_max_ids,
                                                                          word_seq_lens = word_seq_lens,
                                                                          need_to_reverse=True,
                                                                          idx2label=config.ner_idx2labels)
                all_predictions.extend(predict_sequences)
                gold_sequences = from_label_id_tensor_to_label_sequence(batch_ids = batch_label_ids,
                                                                        word_seq_lens = word_seq_lens,
                                                                        need_to_reverse=False,
                                                                        idx2label=config.ner_idx2labels)
                all_golds.extend(gold_sequences)
            else: # span
                print("waiting for completing.")
    if config.parser_mode == PaserModeType.crf:
        results = metric.compute(predictions=all_predictions, references=all_golds, scheme="IOBES")
        for inst, pred_seq in zip(insts, all_predictions):
            inst.prediction = pred_seq
        f1Scores = []
        if print_each_type_metric or config.print_detail_f1 or (config.earlystop_atr == "macro"):
            for key in results:
                precision_key, recall_key, fscore_key = results[key]["precision"]* 100, results[key]["recall"]* 100, results[key]["f1"]* 100
                logger.info(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")
                f1Scores.append(fscore_key)
            if len(f1Scores) > 0:
                logger.info(f"[{name} set Total] Macro F1: {sum(f1Scores) / len(f1Scores):.2f}")

        precision, recall, fscore = results['overall_precision'] * 100, results['overall_recall']* 100, results['overall_f1']* 100
        logger.info(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}")

        if config.earlystop_atr == "macro" and len(f1Scores) > 0:
            fscore = sum(f1Scores) / len(f1Scores)
    else: # span
        # precision =total_correct / (total_predict+1e-10) * 100
        # recall = total_correct / (total_golden + 1e-10) * 100
        # fscore = precision * recall * 2 / (precision + recall + 1e-10)
        # logger.info(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, Micro F1: {fscore:.2f}")
        print("waiting for completing.")
    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="SynDepAT implementation")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    metric = datasets.load_metric("seqeval")
    conf = Config(opt)
    # NER+DP Data Reading
    tokenizer = AutoTokenizer.from_pretrained(conf.embedder_type, add_prefix_space=True)
    logger.info(f"[NER+DP Info] Reading dataset from: \t{conf.nerdp_train_file}\t{conf.nerdp_dev_file}\t{conf.nerdp_test_file}")
    nerdp_train_dataset = TransformersNERDataset(conf.parser_mode, conf.dep_model, conf.nerdp_train_file, tokenizer, number=-1, is_train=True)
    conf.nerdp_label2idx = nerdp_train_dataset.label2idx
    conf.nerdp_idx2labels = nerdp_train_dataset.idx2labels
    conf.deplabel2idx = nerdp_train_dataset.deplabel2idx
    conf.root_dep_label_id = nerdp_train_dataset.root_dep_label_id
    conf.nerdp_label_size = len(nerdp_train_dataset.label2idx)
    nerdp_dev_dataset = TransformersNERDataset(conf.parser_mode, conf.dep_model, conf.nerdp_dev_file, tokenizer,
                                         number=-1, label2idx=nerdp_train_dataset.label2idx,
                                         deplabel2idx=nerdp_train_dataset.deplabel2idx, is_train=False)
    nerdp_test_dataset = TransformersNERDataset(conf.parser_mode, conf.dep_model, conf.nerdp_test_file, tokenizer,
                                          number=-1, label2idx=nerdp_train_dataset.label2idx,
                                          deplabel2idx=nerdp_train_dataset.deplabel2idx, is_train=False)

    num_workers = 8
    conf.label_size = len(train_dataset.label2idx)
    conf.nerdp_max_entity_length = max(max(nerdp_train_dataset.max_entity_length, nerdp_dev_dataset.max_entity_length),
                                 nerdp_test_dataset.max_entity_length)
    nerdp_train_dataloader = DataLoader(nerdp_train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=nerdp_train_dataset.collate_to_max_length)
    nerdp_dev_dataloader = DataLoader(nerdp_dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=nerdp_dev_dataset.collate_to_max_length)
    nerdp_test_dataloader = DataLoader(nerdp_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=nerdp_test_dataset.collate_to_max_length)
    # Only NER Data Reading
    logger.info(f"[Only NER Data Info] Reading dataset from: \t{conf.ner_train_file}\t{conf.ner_dev_file}\t{conf.ner_test_file}")
    ner_train_dataset = TransformersNERDataset(conf.parser_mode, DepModelType.none, conf.ner_train_file, tokenizer, number=-1, is_train=True)
    conf.ner_label2idx = ner_train_dataset.label2idx
    conf.ner_idx2labels = ner_train_dataset.idx2labels
    conf.ner_label_size = len(ner_train_dataset.label2idx)
    ner_dev_dataset = TransformersNERDataset(conf.parser_mode, DepModelType.none, conf.ner_dev_file, tokenizer,
                                         number=-1, label2idx=ner_train_dataset.label2idx, deplabel2idx=None, is_train=False)

    ner_test_dataset = TransformersNERDataset(conf.parser_mode, DepModelType.none, conf.ner_test_file, tokenizer,
                                         number=-1, label2idx=ner_train_dataset.label2idx, deplabel2idx=None, is_train=False)

    conf.ner_max_entity_length = max(max(ner_train_dataset.max_entity_length,  ner_dev_dataset.max_entity_length),
                                 ner_test_dataset.max_entity_length)

    ner_train_dataloader = DataLoader(ner_train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=ner_train_dataset.collate_to_max_length)
    ner_dev_dataloader = DataLoader(ner_dev_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                collate_fn=ner_dev_dataset.collate_to_max_length)
    ner_test_dataloader = DataLoader(ner_test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=ner_test_dataset.collate_to_max_length)

    train_model(conf, conf.num_epochs, nerdp_train_dataloader, nerdp_dev_dataloader, nerdp_test_dataloader,
                ner_train_dataloader, ner_dev_dataloader, ner_test_dataloader)

if __name__ == "__main__":
    main()
