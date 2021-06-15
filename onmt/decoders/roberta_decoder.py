from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaEmbeddings
from onmt.decoders.transformer import TransformerDecoder
import torch.nn as nn
import torch
import numpy as np
import onmt
import copy

MAX_SIZE = 256


def clone_or_share_layer(layer1, layer2, share=False):
    if share:
        layer1.weight, layer1.bias = layer2.weight, layer2.bias
    else:
        layer1.weight, layer1.bias = \
            nn.Parameter(
                layer2.weight.clone()), nn.Parameter(layer2.bias.clone())


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


class RobertaDecoderEmbeddings(nn.Module):
    def __init__(self, roberta_embeddings: RobertaEmbeddings):
        super(RobertaDecoderEmbeddings, self).__init__()
        self.word_lut = roberta_embeddings.word_embeddings
        self.position_embeddings = roberta_embeddings.position_embeddings
        self.token_type_embeddings = roberta_embeddings.token_type_embeddings

        self.LayerNorm = roberta_embeddings.LayerNorm
        self.dropout = roberta_embeddings.dropout
        self.padding_idx = roberta_embeddings.padding_idx
        self.position_embedding_type = 'absolute'
        self.position_ids = roberta_embeddings.position_ids

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
            step=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if step is not None:
            position_ids.fill_(step)

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


class RobertaDecoderLayer(nn.Module):
    def __init__(self, roberta_layer: RobertaLayer, init_context=False, ):
        super(RobertaDecoderLayer, self).__init__()
        num_heads = roberta_layer.attention.self.num_attention_heads
        hidden_size = roberta_layer.attention.self.query.weight.size(0)
        self.init_context = init_context
        self.dropout = roberta_layer.attention.self.dropout.p

        # create self-attention layer
        self.self_attn = onmt.modules.MultiHeadedAttention(
            num_heads, hidden_size, dropout=self.dropout,
        )
        self.self_attn_drop = roberta_layer.attention.output.dropout
        self.self_attn_norm = copy.deepcopy(roberta_layer.attention.output.LayerNorm)

        # initialize self-attention layers with bert weights
        self.self_attn.linear_keys = roberta_layer.attention.self.key
        self.self_attn.linear_values = roberta_layer.attention.self.value
        self.self_attn.linear_query = roberta_layer.attention.self.query
        self.self_attn.final_linear = roberta_layer.attention.output.dense

        # create context-attention layer 1
        self.context_attn = onmt.modules.MultiHeadedAttention(
            num_heads, hidden_size, dropout=self.dropout,
        )
        self.context_attn_drop = roberta_layer.attention.output.dropout
        self.context_attn_norm = copy.deepcopy(roberta_layer.attention.output.LayerNorm)

        if init_context:
            # Initialize context-attention layers with bert weights
            clone_or_share_layer(
                self.context_attn.linear_keys,
                roberta_layer.attention.self.key,
                share=False,
            )

            clone_or_share_layer(
                self.context_attn.linear_values,
                roberta_layer.attention.self.value,
                share=False,
            )

            clone_or_share_layer(
                self.context_attn.linear_query,
                roberta_layer.attention.self.query,
                share=False,
            )

            clone_or_share_layer(
                self.context_attn.final_linear,
                roberta_layer.attention.output.dense,
            )

        self.intermediate = roberta_layer.intermediate
        self.output = roberta_layer.output

        mask = self._get_attn_subsequent_mask(MAX_SIZE)

        self.register_buffer('mask', mask)

    def forward(
            self,
            inputs,
            memory_bank,
            src_pad_mask,
            tgt_pad_mask,
            layer_cache=None,
            step=None,
    ):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`

        """
        dec_mask = None
        if step is None:
            dec_mask = torch.gt(tgt_pad_mask +
                                self.mask[:, :tgt_pad_mask.size(-1),
                                :tgt_pad_mask.size(-1)], 0)

        query, attn = self.self_attn(inputs, inputs, inputs,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     attn_type="self")

        query_norm = self.self_attn_norm(self.self_attn_drop(query) + inputs)

        mid, attn = self.context_attn(memory_bank, memory_bank,
                                      query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      attn_type="context")

        mid_norm = self.context_attn_norm(
            self.context_attn_drop(mid) + query_norm)

        intermediate_output = self.intermediate(mid_norm)
        output = self.output(intermediate_output, mid_norm)

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class RobertaDecoder(TransformerDecoder):
    def __init__(
            self,
            copy_attn,
            vocab_size,
            pad_idx,
            init_context=False,
            alignment_layer=0,
            alignment_heads=0,
            # token_type='A',
    ):
        super(TransformerDecoder, self).__init__(
            d_model=768,
            copy_attn=copy_attn,
            embeddings=None,
            alignment_layer=alignment_layer,
        )

        # basic attributes
        self.decoder_type = 'roberta'
        self.pad_idx = pad_idx
        # self.token_type = token_type
        self.init_context = init_context

        # decoder state
        self.state = {}
        self._copy = copy_attn
        self.config = RobertaConfig(vocab_size=vocab_size)
        roberta = RobertaModel(self.config)
        self.embeddings = RobertaDecoderEmbeddings(roberta.embeddings)
        self.transformer_layers = nn.ModuleList([
            RobertaDecoderLayer(roberta_layer=roberta_layer, init_context=init_context)
            for roberta_layer in roberta.encoder.layer
        ])

    @classmethod
    def from_opt(cls, opt, embeddings=None,):
        return cls(
            copy_attn=opt.copy_attn,
            vocab_size=embeddings.word_lut.weight.size(0),
            pad_idx=embeddings.word_padding_idx,
            init_context=opt.bert_decoder_init_context,
            # token_type=opt.roberta_decoder_token_type,
        )

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None, with_align=None,):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        if step == 0:
            self._init_cache(memory_bank)

        src = self.state["src"]
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_words, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        # [B, 1, T_src]
        src_pad_mask = src_words.data.eq(self.pad_idx).unsqueeze(1)
        # [B, 1, T_tgt]
        tgt_pad_mask = tgt_words.data.eq(self.pad_idx).unsqueeze(1)

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"]["layer_{}".format(i)] \
                if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step
            )

        # Process the result and update the attentions.
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn[:, 0, :, :].transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        return dec_outs, attns

    def initialize_bert(self, roberta_type):
        print(f"Loading pretrained bert {roberta_type}")
        roberta: RobertaModel = RobertaModel.from_pretrained(roberta_type)
        roberta.resize_token_embeddings(self.embeddings.word_lut.num_embeddings)
        self.embeddings = RobertaDecoderEmbeddings(roberta.embeddings)
        self.transformer_layers = nn.ModuleList([
            RobertaDecoderLayer(roberta_layer, self.init_context)
            for roberta_layer in roberta.encoder.layer
        ])

        if not self.init_context:
            for transformer_layer in self.transformer_layers:
                transformer_layer.context_attn.apply(roberta.init_weights)
