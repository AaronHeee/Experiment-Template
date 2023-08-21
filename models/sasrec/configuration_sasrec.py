from transformers import PretrainedConfig

class SASRecConfig(PretrainedConfig):
    model_type = "sasrec"

    def __init__(self, n_items=10, n_layers=2, n_heads=2, hidden_size=64, max_len=50, dropout=0, **kwargs):
        super().__init__(**kwargs)

        self.n_items = n_items
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.max_len = max_len 
        self.dropout = dropout