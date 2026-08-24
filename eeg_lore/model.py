import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Tokenizer(nn.Module):
    def __init__(self, n_channels=65, dim_cnn=64, dim_token=256, window_len=100, pool1=10, dropout=0.5):
        super().__init__()
        if window_len % pool1:
            raise ValueError((window_len, pool1))
        self.n_channels = n_channels
        self.dim_cnn = dim_cnn
        self.dim_token = dim_token
        self.window_len = window_len
        self.pool1 = pool1
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, dim_cnn, (1, 40), padding=(0, 20)),
            nn.Conv2d(dim_cnn, dim_cnn, (n_channels, 1)),
            nn.SyncBatchNorm(dim_cnn),
            nn.GELU(),
            nn.MaxPool2d((1, pool1), stride=(1, pool1)),
            nn.Dropout(dropout),
        )
        self.out_dim = dim_cnn * (window_len // pool1)

    def forward(self, x):
        x = x.unfold(-1, self.window_len, self.window_len)
        batch_size, _, n_windows, _ = x.shape
        x = einops.rearrange(x, "B C N T -> (B N) 1 C T")
        x = self.enc_conv(x)
        return einops.rearrange(x, "(B N) F 1 T -> B N (F T)", B=batch_size, N=n_windows)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
        return x + self._ff_block(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, dim_token, num_layers, ff_scale=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(dim_token, num_heads, dim_token * ff_scale, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_token, elementwise_affine=True)

    def forward(self, tokens, mask=None):
        x = tokens
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.norm(x)


class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_tokens = (config.max_seq_len - config.window_len) // config.window_len + 1
        self.tokenizer = Tokenizer(config.n_channels, config.dim_cnn, config.dim_token, config.window_len)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_tokens + 1, config.dim_token))
        self.freq_cutoff_min = config.freq_cutoff_min
        self.freq_cutoff_max = config.freq_cutoff_max
        self.freq_cutoff_bandwidth = config.freq_cutoff_bandwidth
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.dim_token))
        self.mask_inProj = nn.Linear(self.tokenizer.out_dim, config.dim_token)
        self.mask_transformer = Transformer(
            config.dim_token,
            config.num_layers,
            config.ff_scale,
            config.num_heads,
            config.dropout,
        )
        self.mask_outProj = nn.Sequential(
            nn.Linear(config.dim_token, config.n_channels * 20),
            nn.GELU(),
            Rearrange("B N (C T) -> B N C T", C=config.n_channels),
            nn.Linear(20, config.window_len),
            Rearrange("B N C T -> B C (N T)"),
        )

    @torch.no_grad()
    def frequency_cutoff(self, x):
        low = torch.randint(self.freq_cutoff_min, self.freq_cutoff_max + 1, (1,)).item()
        high = low + self.freq_cutoff_bandwidth
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            spectrum = torch.fft.rfft(x32, dim=-1)
            frequencies = torch.fft.rfftfreq(x32.size(-1), d=1.0 / self.config.sfreq)
            spectrum[..., (frequencies >= low) & (frequencies < high)] = 0
            return torch.fft.irfft(spectrum, n=x.size(-1), dim=-1).type_as(x)

    def reconstruction_loss(self, x):
        corrupted = self.frequency_cutoff(x)
        tokens = self.tokenizer(corrupted)
        batch_size, n_tokens, _ = tokens.shape
        mask = torch.rand((batch_size, n_tokens), device=tokens.device) < self.config.mask_ratio
        encoded = self.mask_inProj(tokens.clone())
        encoded[mask] = self.mask_token.to(encoded.dtype)
        predicted = self.mask_transformer(encoded + self.pos_embedding[:, :n_tokens])
        return F.mse_loss(self.mask_outProj(predicted), x)

    def forward(self, x):
        tokens = self.tokenizer(x)
        n_tokens = tokens.size(1)
        return self.mask_transformer(self.mask_inProj(tokens) + self.pos_embedding[:, :n_tokens])


class MeanPool(nn.Module):
    def forward(self, condition, tokens):
        return tokens.mean(dim=1)


class EEGLoREBase(nn.Module):
    def __init__(self, config, dataset_to_task, n_tasks, num_datasets, prototypes, task_embeddings, dropout=0.1):
        super().__init__()
        self.config = config
        self.n_tasks = n_tasks
        self.register_buffer("prototypes", F.normalize(torch.as_tensor(prototypes), dim=-1).t(), persistent=True)
        self.register_buffer("meta_embs", torch.as_tensor(task_embeddings, dtype=torch.float32), persistent=True)
        self.tower = EEGEncoder(config)
        self.ds_embed = nn.Embedding(num_datasets, config.dim_text_emb)
        nn.init.zeros_(self.ds_embed.weight)
        self.ds_scale = nn.Parameter(torch.tensor(1.0))
        self.task_adaln = nn.ModuleList([
            nn.Linear(config.dim_text_emb, config.dim_token * 2)
            for _ in range(n_tasks)
        ])
        for layer in self.task_adaln:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.task_proj = nn.ModuleList([
            nn.Linear(config.dim_token, config.dim_text_emb)
            for _ in range(n_tasks)
        ])
        self.task_pool = nn.ModuleList([MeanPool() for _ in range(n_tasks)])
        self.register_buffer("ds_to_task", torch.as_tensor(dataset_to_task, dtype=torch.long), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def _condition(self, instruction_embedding, dataset_ids):
        return instruction_embedding + self.ds_scale * self.ds_embed(dataset_ids)

    def _experts(self, tokens, condition, task_ids):
        output = tokens.new_zeros(tokens.size(0), self.config.dim_text_emb)
        for task_index in range(self.n_tasks):
            selected = task_ids == task_index
            if not selected.any():
                continue
            scale, shift = self.task_adaln[task_index](condition[selected]).chunk(2, dim=-1)
            modulated = tokens[selected] * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            pooled = self.task_pool[task_index](condition[selected], modulated)
            output[selected] = self.task_proj[task_index](pooled)
        return output

    def forward(self, x, instruction_embedding, dataset_ids):
        tokens = self.tower(x)
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0).expand(x.size(0), -1)
        condition = self._condition(instruction_embedding, dataset_ids)
        task_ids = self.ds_to_task[dataset_ids]
        output = self._experts(tokens, condition, task_ids)
        return F.normalize(self.dropout(output), dim=-1), tokens

    def get_logits(self, x, instruction_embedding, dataset_ids):
        embedding = self(x, instruction_embedding, dataset_ids)[0]
        return embedding @ self.prototypes

    def route(self, instruction_embedding):
        query = F.normalize(instruction_embedding.float().reshape(1, -1), dim=-1)
        task_embeddings = F.normalize(self.meta_embs, dim=-1)
        return (query @ task_embeddings.t()).argmax(dim=-1).squeeze()

    def direct_logits(self, x, instruction_embedding, dataset_ids):
        task_index = self.route(instruction_embedding).item()
        tokens = self.tower(x)
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0).expand(x.size(0), -1)
        condition = self._condition(instruction_embedding, dataset_ids)
        task_ids = torch.full((x.size(0),), task_index, device=x.device, dtype=torch.long)
        output = self._experts(tokens, condition, task_ids)
        return F.normalize(self.dropout(output), dim=-1) @ self.prototypes


class EEGLoRE(EEGLoREBase):
    def __init__(self, config, dataset_to_task, n_tasks, num_datasets, prototypes, task_embeddings, dropout=0.1):
        super().__init__(config, dataset_to_task, n_tasks, num_datasets, prototypes, task_embeddings, dropout)
        self.proto_adapters = nn.ModuleList([
            nn.Linear(config.dim_text_emb, config.dim_text_emb, bias=False)
            for _ in range(n_tasks)
        ])
        for adapter in self.proto_adapters:
            nn.init.eye_(adapter.weight)

    def adapted_prototypes(self, task_index):
        adapted = self.proto_adapters[task_index](self.prototypes.t())
        return F.normalize(adapted, dim=-1).t()

    def adapter_loss(self):
        total = self.proto_adapters[0].weight.new_zeros(())
        for adapter in self.proto_adapters:
            weight = adapter.weight
            identity = torch.eye(weight.size(0), device=weight.device, dtype=weight.dtype)
            total = total + (weight.t() @ weight - identity).pow(2).sum()
        return total

    def logits_from_embedding(self, embedding, dataset_ids):
        task_ids = self.ds_to_task[dataset_ids]
        logits = embedding.new_zeros(embedding.size(0), self.prototypes.size(1))
        for task_index in range(self.n_tasks):
            selected = task_ids == task_index
            if selected.any():
                logits[selected] = embedding[selected] @ self.adapted_prototypes(task_index)
        return logits

    def get_logits(self, x, instruction_embedding, dataset_ids):
        return self.logits_from_embedding(self(x, instruction_embedding, dataset_ids)[0], dataset_ids)

    def direct_logits(self, x, instruction_embedding, dataset_ids):
        task_index = self.route(instruction_embedding).item()
        tokens = self.tower(x)
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0).expand(x.size(0), -1)
        condition = self._condition(instruction_embedding, dataset_ids)
        task_ids = torch.full((x.size(0),), task_index, device=x.device, dtype=torch.long)
        output = self._experts(tokens, condition, task_ids)
        embedding = F.normalize(self.dropout(output), dim=-1)
        return embedding @ self.adapted_prototypes(task_index)
