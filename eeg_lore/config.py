from dataclasses import dataclass


@dataclass
class ModelConfig:
    sfreq: int = 200
    n_channels: int = 65
    dim_cnn: int = 64
    num_layers: int = 12
    dim_token: int = 256
    max_seq_len: int = 2000
    window_len: int = 100
    ff_scale: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    mask_ratio: float = 0.5
    freq_cutoff_min: int = 1
    freq_cutoff_max: int = 50
    freq_cutoff_bandwidth: int = 6
    dim_text_emb: int = 768
