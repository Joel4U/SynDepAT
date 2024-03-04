# 
# @author: Allan
#

from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import collections
from src.config.config import PaserModeType
import numpy as np
from src.data.data_utils import convert_iobes, build_label_idx, check_all_labels_in_dict, check_all_obj_is_None, build_spanlabel_idx, build_deplabel_idx, build_poslabel_idx
import pickle
from src.data import Instance
from src.data.data_utils import UNK

Feature = collections.namedtuple('Feature', 'words word_seq_len context_emb chars char_seq_lens poslabel dep_head dep_label labels')
Feature.__new__.__defaults__ = (None,) * 9


class NERDataset(Dataset):

    def __init__(self, file: str, parser_mode: int, poslabel2idx: Dict[str, int] = None,
                 label2idx: Dict[str, int] = None, deplabel2idx: Dict[str, int] = None, is_train: bool = True):
        """
        Read the dataset into Instance
        """
        self.parser_mode = parser_mode
        ## read all the instances. sentences and labels
        insts = self.read_file(file=file)
        self.insts = insts
        if is_train:
            if label2idx is not None:
                print(f"[WARNING] YOU ARE USING EXTERNAL label2idx, WHICH IS NOT BUILT FROM TRAINING SET.")
                self.label2idx = label2idx
            else:
                print(f"[Data Info] Using the training set to build label index")
                if parser_mode == PaserModeType.crf:
                    idx2labels, label2idx = build_label_idx(self.insts)
                else:
                    idx2labels, label2idx = build_spanlabel_idx(self.insts)

                self.poslabel2idx, self.idx2poslabel = build_poslabel_idx(self.insts)
                self.idx2labels = idx2labels
                self.label2idx = label2idx
                self.deplabel2idx, self.root_dep_label_id = build_deplabel_idx(self.insts)
        else:
            assert label2idx is not None ## for dev/test dataset we don't build label2idx, pass in label2idx argument
            self.label2idx = label2idx
            check_all_labels_in_dict(insts=insts, label2idx=self.label2idx)
            self.deplabel2idx = deplabel2idx
            self.poslabel2idx = poslabel2idx

    def convert_instances_to_feature_tensors(self, word2idx: Dict[str, int], char2idx: Dict[str, int]):
        self.inst_ids = []
        for i, inst in enumerate(self.insts):
            words = inst.words
            word_ids = []
            char_ids = []
            output_ids = []
            char_seq_lens = []
            pos_labels=[]
            for word in words:
                if word in word2idx:
                    word_ids.append(word2idx[word])
                else:
                    word_ids.append(word2idx[UNK])
                char_id = []
                char_seq_lens.append(len(word))
                for c in word:
                    if c in char2idx:
                        char_id.append(char2idx[c])
                    else:
                        char_id.append(char2idx[UNK])
                char_ids.append(char_id)
            if inst.labels is not None:
                for label in inst.labels:
                    output_ids.append(self.label2idx[label])
            for pos_tag in inst.pos_tags:
                pos_labels.append(self.poslabel2idx[pos_tag])
            deplabel_ids = [self.deplabel2idx[dep_label] for dep_label in inst.dep_labels] if inst.dep_labels else [-100] * len(words)
            dephead_ids = inst.dep_heads
            context_emb = inst.vec
            self.inst_ids.append(Feature(words = word_ids, chars = char_ids, word_seq_len = len(words), char_seq_lens = char_seq_lens,
                                         context_emb = context_emb, pos_labels= pos_labels, dep_head=dephead_ids, dep_label= deplabel_ids,
                                         labels = output_ids if inst.labels is not None else None))

    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        print(f"[Data Info] Reading file: {file}, labels will be converted to IOBES encoding under CRF parsing mode.")
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            dep_heads = []
            dep_labels = []
            ori_words = []
            pos_labels = []
            labels = []
            chunks = []
            find_root = False
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line.startswith("-DOCSTART"):
                    continue
                if line == "" and len(words) != 0:
                    if self.parser_mode == PaserModeType.crf:
                        labels = convert_iobes(labels)
                    else:
                        chunks = self.get_chunks(labels)
                    insts.append(Instance(words=words, ori_words=ori_words, pos_labels=pos_labels, dep_heads=dep_heads, dep_labels=dep_labels, span_labels=chunks, labels=labels))
                    pos_labels = []
                    find_root = False
                    dep_heads = []
                    dep_labels = []
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                elif line == "" and len(words) == 0:
                    continue
                ls = line.split()
                word, tag, head, dep_label, label = ls[1], ls[3], int(ls[6]), ls[7], ls[-1]
                if head == 0 and find_root:
                    raise ValueError("already have a root")
                pos_labels.append(tag)
                dep_heads.append(head - 1) ## because of 0-indexed.
                dep_labels.append(dep_label)
                ori_words.append(word)
                words.append(word)
                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, index):
        return self.inst_ids[index]

    def collate_fn(self, batch:List[Feature]):
        word_seq_lens = [len(feature.words) for feature in batch]
        max_seq_len = max(word_seq_lens)
        max_char_seq_len = -1
        for feature in batch:
            curr_max_char_seq_len = max(feature.char_seq_lens)
            max_char_seq_len = max(curr_max_char_seq_len, max_char_seq_len)
        for i, feature in enumerate(batch):
            padding_length = max_seq_len - len(feature.words)
            words = feature.words + [0] * padding_length
            chars = []
            char_seq_lens = feature.char_seq_lens + [1] * padding_length
            for word_idx in range(feature.word_seq_len):
                pad_char_length = max_char_seq_len - feature.char_seq_lens[word_idx]
                word_chars = feature.chars[word_idx] + [0] * pad_char_length
                chars.append(word_chars)
            for _ in range(max_seq_len - feature.word_seq_len):
                chars.append([0] * max_char_seq_len)
            dephead = feature.dep_head + [0] * padding_length
            deplabel = feature.dep_label + [0] * padding_length
            poslabel = feature.pos_label + [0] * padding_length
            labels = feature.labels + [0] * padding_length if feature.labels is not None else None

            batch[i] = Feature(words=np.asarray(words), chars=np.asarray(chars), char_seq_lens=np.asarray(char_seq_lens),
                               context_emb = feature.context_emb, word_seq_len = feature.word_seq_len,
                               poslabel = poslabel, dephead = dephead, deplabel = deplabel,
                               labels= np.asarray(labels) if labels is not None else None)
        results = Feature(*(default_collate(samples) if not check_all_obj_is_None(samples) else None for samples in zip(*batch) ))
        return results

    def load_elmo_vec(self, file: str, insts):
        """
        Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
        :param file: the vector files for the ELMo vectors
        :param insts: list of instances
        :return:
        """
        f = open(file, 'rb')
        all_vecs = pickle.load(f)  # variables come out in the order you put them in
        f.close()
        size = 0
        import numpy as np
        if 'bert' in file:
            for vec, inst in zip(all_vecs, insts):
                vec = np.squeeze(vec, axis=0)
                inst.vec = vec
                size = vec.shape[1]
                # print(str(vec.shape[0]) + ","+ str(len(inst.input.words)) + ", " + str(inst.input.words))
                assert (vec.shape[0] == len(inst.input.words))
        else:
            for vec, inst in zip(all_vecs, insts):
                inst.vec = vec
                size = vec.shape[1]
                # print(str(vec.shape[0]) + ","+ str(len(inst.input.words)) + ", " + str(inst.input.words))
                assert (vec.shape[0] == len(inst.input.words))
        return size

# ##testing code to test the dataset loader
# train_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=True)
# label2idx = train_dataset.label2idx
# dev_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=False, label2idx=label2idx)
# test_dataset = NERDataset(file="data/conll2003_sample/train.txt",is_train=False, label2idx=label2idx)
#
# word2idx, _, char2idx, _ = build_word_idx(train_dataset.insts, dev_dataset.insts, test_dataset.insts)
# train_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
# dev_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
# test_dataset.convert_instances_to_feature_tensors(word2idx=word2idx, char2idx=char2idx)
#
#
# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn)
# print(len(train_dataloader))
# for batch in train_dataloader:
#     print(batch.words)
#     exit(0)
#     # pass
