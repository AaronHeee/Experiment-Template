import torch
from torch import nn 

from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from typing import Optional

from .configuration_sasrec import SASRecConfig

class SASRecModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None 
    logits: Optional[torch.FloatTensor] = None


class SASRecModel(PreTrainedModel):
    config_class = SASRecConfig

    def __init__(self, config):

        super().__init__(config)
        
        # Here I use GPT2Model as the proxy of SASRec transformer-block implementation,
        # Feel free to replace this model with any model you like.
        # The only requirement is to keep the inputs, outputs for `feat` and `forward` functions the same.

        from transformers import GPT2Config, GPT2Model 

        config = GPT2Config(
            vocab_size=config.n_items+2,
            n_positions=config.max_len,
            n_ctx=config.max_len,
            n_embd=config.hidden_size,
            n_layer=config.n_layers,
            n_head=config.n_heads,
            activation_function='gelu',
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
        )
        self.model = GPT2Model(config)
        self.item_embed = self.model.wte

        # loss
        self.loss = nn.CrossEntropyLoss()

    def feat(self, x, mask):
        inputs = {'input_ids': x, 'attention_mask': mask}
        outputs = self.model(**inputs)
        return outputs[0]

    def forward(self, x, mask, labels=None):
        representation = self.feat(x, mask) # (bs, max_len, hidden_size)
        logits = representation @ self.item_embed.weight.t() # (bs, max_len, hidden_size) @ (hidden_size, vocab_size)
        if labels is None:
            return SASRecModelOutput(loss=None, logits=logits) 
        else:
            loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
            return SASRecModelOutput(loss=loss, logits=logits)
