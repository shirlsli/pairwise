import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import pearsonr

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 attn_dropout, hidden_dropout, topk=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk = topk

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None,
                sparse_flag=False, output_attention=False):
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        q = self.q(x).view(B, L, H, HD).transpose(1, 2)
        k = self.k(x).view(B, L, H, HD).transpose(1, 2)
        v = self.v(x).view(B, L, H, HD).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn + attention_mask

        if sparse_flag and self.topk is not None:
            topk_val, _ = torch.topk(attn, self.topk, dim=-1)
            threshold = topk_val[..., -1].unsqueeze(-1)
            attn = attn.masked_fill(attn < threshold, float('-inf'))

        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        out = self.layer_norm(x + self.hidden_dropout(out))

        if output_attention:
            return out, attn_weights
        return out, None


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads,
                 attn_dropout, hidden_dropout,
                 topk_query=None, topk_key=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_query = topk_query
        self.topk_key = topk_key

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value,
                key_mask=None, query_mask=None,
                sparse_flag=False, output_attention=False):
        B, Lq, D = query.shape
        H, HD = self.num_heads, self.head_dim

        q = self.q(query).view(B, Lq, H, HD).transpose(1, 2)
        k = self.k(key_value).view(B, -1, H, HD).transpose(1, 2)
        v = self.v(key_value).view(B, -1, H, HD).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_mask is not None:
            attn = attn + key_mask

        attn_weights = torch.softmax(attn, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)
        out = self.layer_norm(query + self.hidden_dropout(out))

        if output_attention:
            return out, attn_weights
        return out, None


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out = self.fc2(self.dropout(self.act(self.fc1(x))))
        return self.layer_norm(x + out)


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size,
                 num_heads, attn_dropout, hidden_dropout, topk=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            hidden_size, num_heads, attn_dropout, hidden_dropout, topk
        )
        self.ff = FeedForward(hidden_size, intermediate_size, hidden_dropout)

    def forward(self, x, attention_mask=None,
                sparse_flag=False, output_attention=False):
        x, attn_weights = self.attn(
            x, attention_mask, sparse_flag, output_attention
        )
        x = self.ff(x)
        return x, attn_weights


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size,
                 num_heads, attn_dropout, hidden_dropout,
                 topk_query=None, topk_key=None):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttention(
            hidden_size, num_heads, attn_dropout, hidden_dropout,
            topk_query, topk_key
        )
        self.self_attn = MultiHeadSelfAttention(
            hidden_size, num_heads, attn_dropout, hidden_dropout
        )
        self.ff = FeedForward(hidden_size, intermediate_size, hidden_dropout)

    def forward(self, cell_embed, drug_embed,
                drug_mask=None, cell_mask=None,
                sparse_flag=False, output_attention=False):
        # Cell attends to drug
        cell_embed, attn_weights = self.cross_attn(
            cell_embed, drug_embed,
            key_mask=drug_mask, query_mask=cell_mask,
            sparse_flag=sparse_flag,
            output_attention=output_attention
        )
        # Drug self-attention
        drug_embed, _ = self.self_attn.__class__.forward(
            self.self_attn, drug_embed,
            attention_mask=drug_mask,
            sparse_flag=sparse_flag,
            output_attention=False
        )
        cell_embed = self.ff(cell_embed)
        return cell_embed, drug_embed, attn_weights


class AttnEncoder(nn.Module):
    """
    Flexible encoder supporting interleaved self-attention (SA) and
    cross-attention (CA) layers, controlled by a '+'-delimited structure
    string (e.g. 'SA+SA' for base encoder, 'CA+SA+CA' for pert encoder).
    """
    def __init__(self, hidden_size, intermediate_size,
                 num_heads, attn_dropout, hidden_dropout,
                 topk_cell, topk_drug, structure, sparse_flag=False):
        super().__init__()
        self.structure = structure
        self.layers_spec = structure.split('+')
        self.sparse_flag = sparse_flag

        n_ca = self.layers_spec.count('CA')
        n_sa = self.layers_spec.count('SA')

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                hidden_size, intermediate_size, num_heads,
                attn_dropout, hidden_dropout, topk_cell, topk_drug
            ) for _ in range(n_ca)
        ])
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(
                hidden_size, intermediate_size, num_heads,
                attn_dropout, hidden_dropout, topk_cell
            ) for _ in range(n_sa)
        ])

    def forward(self, cell_embed, drug_embed=None,
                cell_mask=None, drug_mask=None,
                output_attention=False):
        ca_idx = 0
        sa_idx = 0
        attention_dict = {}

        for step, layer_type in enumerate(self.layers_spec):
            if layer_type == 'CA':
                assert drug_embed is not None, \
                    "drug_embed required for CA layers"
                cell_embed, drug_embed, attn = self.cross_blocks[ca_idx](
                    cell_embed, drug_embed,
                    drug_mask=drug_mask, cell_mask=cell_mask,
                    sparse_flag=self.sparse_flag,
                    output_attention=output_attention
                )
                if output_attention:
                    attention_dict[f'CA_{step}'] = attn
                ca_idx += 1
            elif layer_type == 'SA':
                cell_embed, attn = self.self_blocks[sa_idx](
                    cell_embed,
                    attention_mask=cell_mask,
                    sparse_flag=self.sparse_flag,
                    output_attention=output_attention
                )
                if output_attention:
                    attention_dict[f'SA_{step}'] = attn
                sa_idx += 1

        return cell_embed, drug_embed, attention_dict if output_attention else None

class LearnedCellEmbeddings(nn.Module):
    def __init__(self, gene_number, hidden_size, n_bins, dropout):
        super().__init__()
        self.gene_number = gene_number

        self.gene_id_embedding = nn.Embedding(gene_number, hidden_size)
        self.expr_level_embedding = nn.Embedding(n_bins, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.gene_id_embedding.weight, std=0.02)
        nn.init.normal_(self.expr_level_embedding.weight, std=0.02)

    def forward(self, binned_expr):
        batch_size = binned_expr.shape[0]
        gene_indices = (
            torch.arange(self.gene_number, device=binned_expr.device)
            .unsqueeze(0).expand(batch_size, -1)
        )
        gene_id_embed = self.gene_id_embedding(gene_indices)
        expr_embed = self.expr_level_embedding(binned_expr)
        tokens = self.layer_norm(gene_id_embed + expr_embed)
        return self.dropout(tokens)

class MorganDrugEmbeddings(nn.Module):
    def __init__(self, fingerprint_dim, hidden_size,
                 hidden_dim, dropout, num_time_bins):
        super().__init__()

        # Time condition token — discrete embedding over binned durations
        self.time_embedding = nn.Embedding(num_time_bins, hidden_size)

        # Morgan fingerprint → single mol token
        self.fp_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_size)
        )

        self.position_embedding = nn.Embedding(2, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.time_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(self, fingerprint, time_idx):
        time_tok = self.time_embedding(time_idx)    # (batch, hidden_size)
        mol_tok  = self.fp_encoder(fingerprint)     # (batch, hidden_size)

        # Stack into sequence: (batch, 2, hidden_size)
        tokens = torch.stack([time_tok, mol_tok], dim=1)

        # Add positional embeddings
        positions = torch.arange(2, device=fingerprint.device)
        tokens = tokens + self.position_embedding(positions).unsqueeze(0)

        tokens = self.layer_norm(tokens)
        return self.dropout(tokens)

class DeltaHead(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_size, 1)
        )

    def forward(self, trt_embed, ctl_embed):
        diff = trt_embed - ctl_embed
        return self.fc(diff).squeeze(-1)

class XPertMorgan(nn.Module):
    """
    XPert adapted as a transition function fT for SequenTx-style sequential
    drug combination prediction, predicting only delta expression
    (xdeg = xpert - xbase).
    """

    def __init__(
        self,
        gene_number=978,
        fingerprint_dim=1024,
        hidden_size=128,
        n_bins=64,
        num_heads=4,
        ctl_structure='SA+SA',
        trt_structure='CA+SA+CA',
        intermediate_size=None,
        attn_dropout=0.1,
        hidden_dropout=0.1,
        topk_cell=None,
        topk_drug=None,
        drug_hidden_dim=512,
        latent_size=64,
        num_time_bins=6,
        mse_weight=1.0,
        pcc_weight=1.0,
        learning_rate=4e-3,
        weight_decay=1e-5,
        epoch=500,
        device='cuda',
        model_file='xpert_morgan.pt'
    ):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * 2

        self.gene_number = gene_number
        self.epoch = epoch
        self.my_device = torch.device(device)
        self.learning_rate = learning_rate
        self.model_file = model_file
        self.mse_weight = mse_weight
        self.pcc_weight = pcc_weight
        
        self.cell_emb = LearnedCellEmbeddings(
            gene_number=gene_number,
            hidden_size=hidden_size,
            n_bins=n_bins,
            dropout=hidden_dropout
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.drug_emb = MorganDrugEmbeddings(
            fingerprint_dim=fingerprint_dim,
            hidden_size=hidden_size,
            hidden_dim=drug_hidden_dim,
            dropout=hidden_dropout,
            num_time_bins=num_time_bins
        )
        self.base_encoder = AttnEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            topk_cell=topk_cell,
            topk_drug=topk_drug,
            structure=ctl_structure
        )
        self.pert_encoder = AttnEncoder(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            hidden_dropout=hidden_dropout,
            topk_cell=topk_cell,
            topk_drug=topk_drug,
            structure=trt_structure
        )
        self.delta_head = DeltaHead(
            hidden_size=hidden_size,
            latent_size=latent_size
        )
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self._warmup_epochs = max(1, int(0.1 * epoch))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda
        )

        self.to(self.my_device)
        self._init_weights()

    def _lr_lambda(self, epoch_idx):
        if epoch_idx < self._warmup_epochs:
            return float(epoch_idx + 1) / float(self._warmup_epochs)
        progress = (epoch_idx - self._warmup_epochs) / max(
            1, self.epoch - self._warmup_epochs
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pcc_loss(self, pred, target):
        """
        1 - mean sample-level PCC (XPert Ldeg equation 14, gamma term).
        pred, target: (batch, gene_number)
        """
        pred_mean = pred.mean(dim=1, keepdim=True)
        tgt_mean  = target.mean(dim=1, keepdim=True)
        pred_c = pred   - pred_mean
        tgt_c  = target - tgt_mean
        cov = (pred_c * tgt_c).sum(dim=1)
        std = (
            pred_c.pow(2).sum(dim=1).sqrt()
            * tgt_c.pow(2).sum(dim=1).sqrt()
            + 1e-8
        )
        return 1.0 - (cov / std).mean()

    def loss_fn(self, pred, target):
        """
        Ldeg = beta * MSE + gamma * (1 - PCC)  [XPert equation 14]
        """
        mse   = nn.functional.mse_loss(pred, target)
        pcc_l = self._pcc_loss(pred, target)
        return self.mse_weight * mse + self.pcc_weight * pcc_l

    def _unpack_batch(self, batch):
        fingerprint = batch[0].to(self.my_device)
        binned_expr = batch[1].to(self.my_device)
        time_idx    = batch[2].to(self.my_device)
        y           = batch[3].to(self.my_device)
        return fingerprint, binned_expr, time_idx, y

    def forward(self, fingerprint, binned_expr, time_idx,
                output_attention=False):
        batch_size = fingerprint.shape[0]

        gene_tokens = self.cell_emb(binned_expr)
        cls = self.cls_token.expand(batch_size, -1, -1)
        cell_input = torch.cat([cls, gene_tokens], dim=1)
        drug_tokens = self.drug_emb(fingerprint, time_idx)
        ctl_out, _, ctl_attn = self.base_encoder(
            cell_input,
            drug_embed=None,
            output_attention=output_attention
        )
        ctl_gene_embed = ctl_out[:, 1:, :]
        trt_out, _, trt_attn = self.pert_encoder(
            cell_input,
            drug_embed=drug_tokens,
            output_attention=output_attention
        )
        trt_gene_embed = trt_out[:, 1:, :]
        delta = self.delta_head(trt_gene_embed, ctl_gene_embed)

        if output_attention:
            return delta, {'trt': trt_attn, 'ctl': ctl_attn}
        return delta

    def fit(self, train_loader, valid_loader):
        """
        Train with early stopping (patience=50, matching XPert).
        Best model by validation loss is saved to self.model_file.
        Scheduler steps once per epoch after each full pass over training data.
        """
        min_valid_loss   = float('inf')
        patience_counter = 0
        patience = 50

        for epoch_idx in range(self.epoch):

            self.train()
            train_loss = 0.0
            for batch in train_loader:
                fingerprint, binned_expr, time_idx, y = \
                    self._unpack_batch(batch)

                self.optimizer.zero_grad()
                pred = self.forward(fingerprint, binned_expr, time_idx)
                loss = self.loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.scheduler.step()

            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    fingerprint, binned_expr, time_idx, y = \
                        self._unpack_batch(batch)
                    pred = self.forward(fingerprint, binned_expr, time_idx)
                    valid_loss += self.loss_fn(pred, y).item()

            valid_loss /= len(valid_loader)

            if valid_loss < min_valid_loss:
                min_valid_loss   = valid_loss
                patience_counter = 0
                torch.save(self, self.model_file)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch_idx}')
                    break

            print(
                f'Epoch {epoch_idx:4d} | '
                f'train {train_loss:.4f} | '
                f'valid {valid_loss:.4f} | '
                f'lr {self.scheduler.get_last_lr()[0]:.2e}'
            )

    def predict(self, fingerprint, binned_expr, time_idx,
                output_attention=False):
        self.eval()

        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, device=self.my_device, dtype=dtype)
            return x.to(self.my_device)

        fingerprint = to_tensor(fingerprint, torch.float32)
        binned_expr = to_tensor(binned_expr, torch.long)
        time_idx    = to_tensor(time_idx,    torch.long)

        with torch.no_grad():
            if output_attention:
                delta, attn = self.forward(
                    fingerprint, binned_expr, time_idx,
                    output_attention=True
                )
                return delta.cpu().numpy(), attn
            delta = self.forward(fingerprint, binned_expr, time_idx)
            return delta.cpu().numpy()

class PerturbationDataset(Dataset):
    def __init__(self, fingerprints, binned_expr, time_idx, y):
        self.fingerprints = torch.tensor(fingerprints, dtype=torch.float32)
        self.binned_expr  = torch.tensor(binned_expr,  dtype=torch.long)
        self.time_idx     = torch.tensor(time_idx,     dtype=torch.long)
        self.y            = torch.tensor(y,            dtype=torch.float32)

    def __len__(self):
        return len(self.fingerprints)

    def __getitem__(self, idx):
        return (
            self.fingerprints[idx],
            self.binned_expr[idx],
            self.time_idx[idx],
            self.y[idx]
        )

def evaluate_model(model, data_loader):
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            fingerprint, binned_expr, time_idx, y = \
                model._unpack_batch(batch)
            pred = model.forward(fingerprint, binned_expr, time_idx)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    pccs = [
        pearsonr(all_targets[i], all_preds[i])[0]
        for i in range(len(all_targets))
    ]

    print(f'Mean PCC:   {np.mean(pccs):.4f}')
    print(f'Median PCC: {np.median(pccs):.4f}')
    print(f'Std PCC:    {np.std(pccs):.4f}')

    return np.array(pccs)

def compute_bin_edges(train_expr, n_bins=64):
    flat = train_expr.flatten()
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(flat, percentiles)


def apply_binning(expr, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(expr, bin_edges[1:-1])
    return np.clip(binned, 0, n_bins - 1).astype(np.int64)