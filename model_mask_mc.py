import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MultipleChoiceModelOutput, MaskedLMOutput


class BertForQAandMLM(BertPreTrainedModel):

    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids0=None,
        input_ids1=None,
        attention_mask0=None,
        attention_mask1=None,
        token_type_ids0=None,
        token_type_ids1=None,
        answer_labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_labels0=None,
        mask_labels1=None,
        mask_inputs0=None,
        mask_inputs1=None,
        qa=True
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = torch.cat([input_ids0, input_ids1], dim=0)
        attention_mask = torch.cat([attention_mask0, attention_mask1], dim=0)
        token_type_ids = torch.cat([token_type_ids0, token_type_ids1], dim=0)
        mask_labels = torch.cat([mask_labels0, mask_labels1], dim=0)
        mask_inputs = torch.cat([mask_inputs0, mask_inputs1], dim=0)

        if qa:
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
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            batch_size2, _ = logits.size()
            logits0, logits1 = torch.split(logits, [batch_size2 // 2, batch_size2 // 2], dim=0)
            reshaped_logits = torch.cat([logits0, logits1], dim=1)

            loss = None
            if answer_labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, answer_labels)

            if not return_dict:
                output = (reshaped_logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return MultipleChoiceModelOutput(
                loss=loss,
                logits=reshaped_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            outputs = self.bert(
                mask_inputs,
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
            prediction_scores = self.cls(sequence_output)

            masked_lm_loss, loss = None, None
            if mask_labels is not None:
                loss_fct = CrossEntropyLoss(reduction="none")  # -100 index = padding token
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mask_labels.view(-1))
                # classification
                masked_lm_loss = masked_lm_loss.reshape(-1, 512)
                mask = (mask_labels != -100).long()
                logits = (masked_lm_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
                logits = logits.unsqueeze(dim=1)
                batch_size2, _ = logits.size()
                logits0, logits1 = torch.split(logits, [batch_size2 // 2, batch_size2 // 2], dim=0)
                reshaped_logits = torch.cat([logits0, logits1], dim=1)
                mask = logits0.bool().long()
                loss = (loss_fct(reshaped_logits, answer_labels) * mask).sum() / (mask.sum() + 1e-12)

            if not return_dict:
                output = (prediction_scores,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return MaskedLMOutput(
                loss=loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

