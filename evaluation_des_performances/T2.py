"""
This file contains all classes necessary to instanciate a T2 model.
Currently works with CamembertModel models
"""

from typing import Optional, Tuple, Union

import torch
from datasets import Dataset
# from .time2vec import Time2Vec
from time2vec import Time2Vec
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import BatchSampler
from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    CamembertConfig,
    CamembertModel,
    CamembertPreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class T2Dataset(Dataset):
    def __init__(self, arrow_table, info=None, split=None, indices_table=None, fingerprint=None, n_cr=None):
        super().__init__(arrow_table, info=info, split=split, indices_table=indices_table, fingerprint=fingerprint)
        if n_cr is None:
            raise RuntimeError("n_cr was given as None for T2Dataset, please add a value")
        self.n_cr = n_cr

    def __getitem__(self, key):
        # print(f"__getitem__ on {key}")
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, list):
            return self.__getitems__(key)
        return super().__getitem__(list(range(key - self.n_cr, key + 1)))

    def __getitems__(self, keys):
        # print(f"__getitems__ on {keys}")
        d = self.__getitem__(keys.pop(0))
        for key in keys:
            d2 = super().__getitem__(key)
            for k in d2.keys():
                d[k].append(d2[k])
        return d


class T2BatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, n_past_cr) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.n_past_cr = n_past_cr  # number of past cr observed

    # iter for T2Dataset
    def __iter__(self):
        #     # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


class T2PatientBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, n_past_cr, last_indices) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.n_past_cr = n_past_cr  # number of past cr observed
        self.last_indices = last_indices

    # iter for T2Dataset, grouping patients
    def __iter__(self):
        #     # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            i = 0
            index_to_watch = self.last_indices[i]
            current_step = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx == index_to_watch:
                    # last ehr of this patient
                    yield batch[:idx_in_batch]
                    idx_in_batch = 0
                    i += 1
                    index_to_watch = self.last_indices[i] if i < len(self.last_indices) else None
                    batch = [0] * self.batch_size
                    current_step += 1
                elif idx_in_batch == self.batch_size:
                    yield batch
                    current_step += 1
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
                current_step += 1


class CamembertT2Config(CamembertConfig):
    def __init__(self, time_dim: int = 8, nhead: int = 8, num_layers=4, n_cr=3, **kwargs) -> None:
        self.time_dim = time_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.n_cr = n_cr
        super().__init__(**kwargs)


class BertT2Config(BertConfig):
    def __init__(self, time_dim: int = 8, nhead: int = 8, num_layers=4, n_cr=3, **kwargs) -> None:
        self.time_dim = time_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.n_cr = n_cr
        super().__init__(**kwargs)


class T2ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.hidden_size + config.time_dim
        self.n_cr = config.n_cr
        self.t2v = Time2Vec(config.time_dim)
        self.num_labels = config.num_labels
        self.nhead = config.nhead 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=config.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers)

        self.dense = nn.Linear(self.d_model, self.d_model)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.d_model, config.num_labels)

    def forward(self, features, dt, ipp_id):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        time_encoding = self.t2v(dt)
        # mask here instead
        x = torch.cat([x, time_encoding], dim=1).reshape(-1, self.d_model).unfold(0, self.n_cr + 1, 1).swapaxes(1, 2)
        try:
            last_values = ipp_id.unfold(0, self.n_cr + 1, 1).squeeze(1)[:, -1]
        except Exception as exception:
            print(exception)
            print(ipp_id)
            print(ipp_id.shape)
        padding_mask = ipp_id.unfold(0, self.n_cr + 1, 1).squeeze(1).T.ne(last_values).T
        text_mask = padding_mask.unsqueeze(-1).tile(dims=(1, 1, self.d_model))
        x = x.masked_fill(text_mask, 0.0).reshape(-1, self.n_cr + 1, self.d_model)  # pad to zero illegal sequences
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)[:, -1, :]  # last ehr taken
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

        # x = torch.cat([x, time_encoding], dim=1).reshape(self.d_model,-1).unfold(0, self.n_cr + 1, 1).swapaxes(1, 2)
        # padding_mask = ipp_id.unfold(1, self.n_cr + 1, 1).squeeze(1).T.ne(last_values).T

class BertT2ForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = T2ClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        ipp_id: Optional[torch.LongTensor] = None,
        dt: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output, dt, ipp_id)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CamembertT2ForSequenceClassification(CamembertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = CamembertModel(config, add_pooling_layer=False)
        self.classifier = T2ClassificationHead(config)
        self.post_init()

    @classmethod
    def from_pretrained_T2(cls, model_name_or_path: str):
        t2_config = CamembertT2Config.from_pretrained(model_name_or_path)
        return cls.from_pretrained(model_name_or_path, config=t2_config)

    @classmethod
    def from_pretrained_body(
        cls,
        model_name_or_path: str,
        time_dim=None,
        nhead=None,
        num_layers=None,
        n_cr=None,
        problem_type="multi_label_classification",
        id2label=None,
        label2id=None,
    ):
        t2_config = CamembertT2Config.from_pretrained(
            model_name_or_path,
            time_dim=time_dim,
            nhead=nhead,
            num_layers=num_layers,
            n_cr=n_cr,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            problem_type=problem_type,
        )
        return cls.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=True,
            config=t2_config,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        ipp_id: Optional[torch.LongTensor] = None,
        dt: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, dt, ipp_id)
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
