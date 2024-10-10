#
# @author: Allan
#

# from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.linear_crf_inferencer import LinearCRF
from src.model.module.linear_encoder import LinearEncoder
from src.model.embedder import TransformersEmbedder
from src.model.module.deplabel_gcn import DepLabeledGCN
from src.model.module.bilstm_encoder import BiLSTMEncoder
from src.model.module.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from typing import Tuple, Union
from src.config.config import PaserModeType
from torch.nn import functional
from src.model.module.spanextractor import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from src.model.module.TEtransformer import *
from src.model.module.gate_fusion import FusionModule
from src.model.module.adv_loss import Adversarial_loss

_dim_map = {
    "concat": 1,
    "gate-concat": 2,
}

NERDP_Task = True
NER_Task = False

class SynDepAT(nn.Module):

    def __init__(self, config):
        super(SynDepAT, self).__init__()
        self.device = config.device
        # self.dep_model = config.dep_model
        # NER + DP encoder and decoder
        self.nerdp_transformer = TransformersEmbedder(transformer_model_name=config.embedder_type, is_freezing=config.embedder_freezing)
        self.nerdp_transformer_drop = nn.Dropout(config.dropout)
        self.gcn = DepLabeledGCN(config, config.gcn_outputsize, self.nerdp_transformer.get_output_dim(),
                                 self.nerdp_transformer.get_output_dim())
        self.nerdp_label_size = config.nerdp_label_size
        self.ner_label_size = config.ner_label_size
        self.nerdp_parser_mode = config.nerdp_parser_mode
        self.ner_parser_mode = config.ner_parser_mode
        self.root_dep_label_id = config.root_dep_label_id
        self.nerdp_line_encoder = LinearEncoder(label_size=config.nerdp_label_size, input_dim=config.gcn_outputsize)
        if self.nerdp_parser_mode == PaserModeType.crf:
            self.nerdp_crf = LinearCRF(label_size=config.nerdp_label_size, label2idx=config.nerdp_label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                        idx2labels=config.nerdp_idx2labels)
        else:
            #  bucket_widths: Whether to bucket the span widths into log-space buckets. If `False`, the raw span widths are used.
            self.nerdp_endpoint_span_extractor = EndpointSpanExtractor(config.gcn_outputsize * 2,
                                                                  combination='x,y',
                                                                  num_width_embeddings=32,
                                                                  span_width_embedding_dim=50,
                                                                  bucket_widths=True)
            self.nerdp_attentive_span_extractor = SelfAttentiveSpanExtractor(config.gcn_outputsize)
            input_dim = self.nerdp_endpoint_span_extractor.get_output_dim() + self.nerdp_attentive_span_extractor.get_output_dim()
            self.nerdp_span_classifier = MultiNonLinearClassifier(input_dim, config.nerdp_label_size, 0.2) # model_dropout = 0.2
        # NER encoder and decoder
        self.ner_transformer = TransformersEmbedder(transformer_model_name=config.embedder_type, is_freezing = config.embedder_freezing)
        self.ner_transformer_drop = nn.Dropout(config.dropout)
        self.linear = nn.Linear(self.ner_transformer.get_output_dim(), config.gcn_outputsize)
        self.ner_encoder = BiLSTMEncoder(label_size=config.ner_label_size,
                                input_dim=self.ner_transformer.get_output_dim(),
                                hidden_dim=config.gcn_outputsize,
                                drop_lstm=config.dropout)
        # self.ner_encoder = nn.LSTM(self.ner_transformer.get_output_dim(), config.gcn_outputsize // 2, num_layers=1, dropout=0.5, batch_first=True, bidirectional=True)
        self.ner_line_encoder = LinearEncoder(label_size=config.ner_label_size, input_dim=config.gcn_outputsize * 2)
        if self.ner_parser_mode == PaserModeType.crf:
            self.ner_crf = LinearCRF(label_size=config.ner_label_size, label2idx=config.ner_label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                        idx2labels=config.ner_idx2labels)
        else:
            #  bucket_widths: Whether to bucket the span widths into log-space buckets. If `False`, the raw span widths are used.
            self.ner_endpoint_span_extractor = EndpointSpanExtractor(config.gcn_outputsize * 2,
                                                                  combination='x,y',
                                                                  num_width_embeddings=32,
                                                                  span_width_embedding_dim=50,
                                                                  bucket_widths=True)
            self.ner_attentive_span_extractor = SelfAttentiveSpanExtractor(self.ner_transformer.get_output_dim())
            input_dim = self.ner_endpoint_span_extractor.get_output_dim() + self.ner_attentive_span_extractor.get_output_dim()
            self.ner_span_classifier = MultiNonLinearClassifier(input_dim, config.ner_label_size, 0.2) # model_dropout = 0.2
        self.classifier = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        # share encoder
        print("[Model Info] Before input the shared encoding layer, Hidden Size is Dep Encoding layer gcn_outputsize: {} ".format(config.gcn_outputsize))
        final_hidden_dim = config.gcn_outputsize
        self.shared_encoder = TETransformerEncoder(num_layers=config.attn_layer, d_model=final_hidden_dim, output_dim=final_hidden_dim, n_head=8,
                                                   feedforward_dim=2 * final_hidden_dim, dropout=0.3, after_norm=True, attn_type=config.shared_enc_type,
                                                   scale=False, dropout_attn=None, pos_embed=None)
        self.output_dim = final_hidden_dim * _dim_map[config.fusion_type]
        self.fusion_nerdp = FusionModule(fusion_type=config.fusion_type, layer=0, input_size=final_hidden_dim,
                                   output_size=self.output_dim, dropout=config.fusion_dropout)
        self.fusion_ner = FusionModule(fusion_type=config.fusion_type, layer=0, input_size=final_hidden_dim,
                                   output_size=self.output_dim, dropout=config.fusion_dropout)
        self.fc_dropout = nn.Dropout(0.4)
        print("[Model Info] Before input the private decoding layer Hidden Size: {}".format(final_hidden_dim * 2))
        # adversiral layer
        self.adv_loss = Adversarial_loss(final_hidden_dim, config.advloss_dropout)


    def forward(self, Task: bool, subword_input_ids: torch.Tensor, word_seq_lens: torch.Tensor, orig_to_tok_index: torch.Tensor,
                    attention_mask: torch.Tensor, depheads: torch.Tensor, deplabels: torch.Tensor,
                    all_span_lens: torch.Tensor,  all_span_ids: torch.Tensor, all_span_weight:torch.Tensor, real_span_mask: torch.Tensor,
                    labels: torch.Tensor = None,
                    is_train: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bz, _ = subword_input_ids.size()
        max_seq_len = word_seq_lens.max()
        if Task == NERDP_Task:
            word_rep = self.nerdp_transformer(subword_input_ids, orig_to_tok_index, attention_mask)
            word_rep = self.nerdp_transformer_drop(word_rep)
            # dggcn encoding layer
            adj_matrixs = [head_to_adj(max_seq_len, orig_to_tok_index[i], depheads[i]) for i in range(bz)]
            adj_matrixs = np.stack(adj_matrixs, axis=0)
            adj_matrixs = torch.from_numpy(adj_matrixs)
            dep_label_adj = [head_to_adj_label(max_seq_len, orig_to_tok_index[i], depheads[i], deplabels[i], self.root_dep_label_id) for i
                             in range(bz)]
            dep_label_adj = torch.from_numpy(np.stack(dep_label_adj, axis=0)).long()
            private_feature_out = self.gcn(word_rep, word_seq_lens, adj_matrixs, dep_label_adj)
            # shared encoding layer using TENER
            batch_size = word_rep.size(0)
            sent_len = word_rep.size(1)
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=word_rep.device).view(1, sent_len).expand(batch_size, sent_len)
            mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
            shared_feature_out = self.shared_encoder(private_feature_out, mask)
            concat_feature_out = self.fusion_nerdp(private_feature_out, shared_feature_out)
            concat_feature_out = self.fc_dropout(concat_feature_out)
            # adv loss
            task_label_1 = torch.ones((bz, 2), device=self.device, dtype=torch.int8)
            adv_loss = self.adv_loss(shared_feature_out, task_label_1)
            # private decoding
            if self.nerdp_parser_mode == PaserModeType.crf:
                encoder_scores = self.nerdp_line_encoder(concat_feature_out, word_seq_lens)
                if is_train:
                    unlabed_score, labeled_score = self.nerdp_crf(encoder_scores, word_seq_lens, labels, mask)
                    return unlabed_score - labeled_score + adv_loss
                else:
                    bestScores, decodeIdx = self.nerdp_crf.decode(encoder_scores, word_seq_lens)
                    return decodeIdx
            else: # span and hidden states begin and end private_feature_out
                all_span_rep = self.nerdp_endpoint_span_extractor(concat_feature_out, all_span_ids.long(), real_span_mask)  # [batch, n_span, hidden]
                att_span_emb = self.nerdp_attentive_span_extractor(private_feature_out, all_span_ids.long(), real_span_mask)
                all_span_rep = torch.cat((all_span_rep, att_span_emb), dim=-1)
                all_span_rep = self.nerdp_span_classifier(all_span_rep)  # (batch,n_span,n_class)
                if is_train:
                    _, n_span = labels.size()
                    all_span_rep = all_span_rep.view(-1, self.nerdp_label_size)
                    span_label = labels.view(-1)
                    loss = self.cross_entropy(all_span_rep, span_label)
                    loss = loss.view(bz, n_span) * all_span_weight
                    loss = torch.masked_select(loss, real_span_mask.bool())
                    return torch.mean(loss) + adv_loss
                else:
                    predicts = self.classifier(all_span_rep)
                    return predicts
        elif Task == NER_Task:
            word_rep = self.ner_transformer(subword_input_ids, orig_to_tok_index, attention_mask)
            word_rep = self.ner_transformer_drop(word_rep)
            # private_feature_out = self.linear(word_rep)
            private_feature_out = self.ner_encoder(word_rep, word_seq_lens)
            # shared encoding layer using TENER
            batch_size = word_rep.size(0)
            sent_len = word_rep.size(1)
            maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=word_rep.device).view(1, sent_len).expand(
                batch_size, sent_len)
            mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
            shared_feature_out = self.shared_encoder(private_feature_out, mask)
            concat_feature_out = self.fusion_ner(private_feature_out, shared_feature_out)
            concat_feature_out = self.fc_dropout(concat_feature_out)
            # adv loss
            task_label_0 = torch.zeros((bz, 2), device=self.device, dtype=torch.int8)
            adv_loss = self.adv_loss(shared_feature_out, task_label_0)
            # private decoding
            if self.ner_parser_mode == PaserModeType.crf:
                encoder_scores = self.ner_line_encoder(concat_feature_out, word_seq_lens)
                if is_train:
                    unlabed_score, labeled_score = self.ner_crf(encoder_scores, word_seq_lens, labels, mask)
                    return unlabed_score - labeled_score + adv_loss
                else:
                    bestScores, decodeIdx = self.ner_crf.decode(encoder_scores, word_seq_lens)
                    return decodeIdx
            else: # span and hidden states begin and end tcat
                all_span_rep = self.ner_endpoint_span_extractor(concat_feature_out, all_span_ids.long(), real_span_mask)  # [batch, n_span, hidden]
                att_span_emb = self.ner_attentive_span_extractor(word_rep, all_span_ids.long(), real_span_mask)
                all_span_rep = torch.cat((all_span_rep, att_span_emb), dim=-1)
                all_span_rep = self.ner_span_classifier(all_span_rep)  # (batch,n_span,n_class)
                if is_train:
                    _, n_span = labels.size()
                    all_span_rep = all_span_rep.view(-1, self.ner_label_size)
                    span_label = labels.view(-1)
                    loss = self.cross_entropy(all_span_rep, span_label)
                    loss = loss.view(bz, n_span) * all_span_weight
                    loss = torch.masked_select(loss, real_span_mask.bool())
                    return torch.mean(loss) + adv_loss
                else:
                    predicts = self.classifier(all_span_rep)
                    return predicts

