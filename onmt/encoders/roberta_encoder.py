from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
import torch.nn as nn
import torch
from torch import Tensor, device
from typing import Tuple, Optional

from onmt.encoders.encoder import EncoderBase


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class RobertaEncoderEmbeddings(nn.Module):
    def __init__(self, roberta_embeddings: RobertaEmbeddings):
        super(RobertaEncoderEmbeddings, self).__init__()
        self.word_lut = roberta_embeddings.word_embeddings
        self.position_embeddings = roberta_embeddings.position_embeddings
        self.token_type_embeddings = roberta_embeddings.token_type_embeddings

        self.LayerNorm = roberta_embeddings.LayerNorm
        self.dropout = roberta_embeddings.dropout
        self.padding_idx = roberta_embeddings.padding_idx
        self.position_embedding_type = 'absolute'
        self.position_ids = roberta_embeddings.position_ids

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_lut(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class RobertaEncoder(EncoderBase):
    """
    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, vocab_size, pad_idx):
        super(RobertaEncoder, self).__init__()
        self.config = RobertaConfig(vocab_size=vocab_size, pad_token_id=pad_idx,)
        roberta = RobertaModel(self.config)
        self.embeddings = RobertaEncoderEmbeddings(roberta.embeddings)
        self.encoder = roberta.encoder

        self.pad_idx = pad_idx

    @classmethod
    def from_opt(cls, opt, embeddings=None,):
        """Alternate constructor."""
        return cls(
            embeddings.word_lut.weight.size(0),
            embeddings.word_padding_idx
        )

    def forward(self, src, lengths=None, **kwargs):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # bert receives a tensor of shape [batch_size x src_len]
        input_ids = src[:, :, 0].t()
        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = (input_ids != self.pad_idx).type(torch.int)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        last_hidden_state = encoder_outputs[0]

        return embedding_output, last_hidden_state.transpose(0, 1), lengths

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            # if self.config.is_decoder:
            #     batch_size, seq_length = input_shape
            #     seq_ids = torch.arange(seq_length, device=device)
            #     causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            #     # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            #     # causal and attention masks must have same type with pytorch version < 1.3
            #     causal_mask = causal_mask.to(attention_mask.dtype)
            #
            #     if causal_mask.shape[1] < attention_mask.shape[1]:
            #         prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            #         causal_mask = torch.cat(
            #             [
            #                 torch.ones(
            #                     (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
            #                 ),
            #                 causal_mask,
            #             ],
            #             dim=-1,
            #         )
            #
            #     extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            # else:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_lut.weight.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def initialize_bert(self, bert_type):
        print(f"Loading pretrained bert {bert_type}")
        roberta: RobertaModel = RobertaModel.from_pretrained(bert_type)
        roberta.resize_token_embeddings(self.embeddings.word_lut.num_embeddings)

        self.embeddings = RobertaEncoderEmbeddings(roberta.embeddings)

        self.encoder = roberta.encoder
