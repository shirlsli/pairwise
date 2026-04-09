import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

"""
Guo, Y., Zhang, H., Hu, H. et al. 
Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer. 
Nat Mach Intell 8, 96–112 (2026). https://doi.org/10.1038/s42256-025-01165-w
"""
class PerturbationDataset(Dataset):
    def __init__(self, split_data, cell_id_to_idx):
        self.drug_fp      = torch.tensor(split_data['drug_fp'],      dtype=torch.float32)
        self.ccl          = torch.tensor(split_data['ccl'],          dtype=torch.float32)
        self.ccl_binned   = torch.tensor(split_data['ccl_binned'],   dtype=torch.long)
        self.delta_expr   = torch.tensor(split_data['delta_expr'],   dtype=torch.float32)
        self.time_idx     = torch.tensor(split_data['time_idx'],     dtype=torch.long)
        self.cell_ids     = split_data['cell_ids']

        self.cell_idx = torch.tensor(
            [cell_id_to_idx[c] for c in self.cell_ids], dtype=torch.long
        )

    def __len__(self):
        return len(self.drug_fp)

    def __getitem__(self, idx):
        # trt_raw_data and ctl_raw_data are the same here since we only
        # have the control expression and the delta — model needs both
        return (
        self.delta_expr[idx],    # trt_raw_data
        self.ccl[idx],           # ctl_raw_data
        self.ccl_binned[idx],    # ctl_raw_data_binned
        self.drug_fp[idx],       # drug_feat
        self.time_idx[idx],      # pert_time_idx
        self.cell_idx[idx],      # cell_idx
        )


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class cell_Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate, pretrained_embed_path):
        super(cell_Embeddings, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)

        pretrained_embed = np.load(pretrained_embed_path, allow_pickle=True)
        pretrained_embed = torch.tensor(pretrained_embed).float()
        if hidden_size != pretrained_embed.size(-1):
            self.linear = nn.Linear(pretrained_embed.size(-1), hidden_size)
            pretrained_embed = self.linear(pretrained_embed)

        self.pretrained_embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        words_embeddings = self.word_embeddings(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        pretrained_embed = self.pretrained_embed(position_ids)
        embeddings = words_embeddings + pretrained_embed

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class morgan_Embeddings(nn.Module):
    def __init__(self, hidden_size, dropout_rate, time_embed):
        super(morgan_Embeddings, self).__init__()
        self.time_embed = time_embed
        self.position_embeddings = nn.Embedding(2, hidden_size)  # time + mol
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(1024, hidden_size)

    def forward(self, input_embed, pert_time_idx):
        mol_embed = self.linear(input_embed).unsqueeze(1)
        time_embed = self.time_embed(pert_time_idx).unsqueeze(1)
        input_embeddings = torch.cat([time_embed, mol_embed], dim=1)

        seq_length = input_embeddings.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embed.device)
        position_ids = position_ids.unsqueeze(0).expand(input_embeddings.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout_p = attention_probs_dropout_prob

    def forward(self, hidden_states, attention_mask=None, output_attention=False):
        batch_size, seq_len, hidden_size = hidden_states.size()

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        if output_attention:
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_probs, value)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            return context, attention_probs
        else:
            context = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
            return context, None


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout_p = attention_probs_dropout_prob

    def forward(self, cell, drug, cell_attention_mask=None,
            drug_attention_mask=None, output_attention=False):
        batch_size, seq_len_A, hidden_size = cell.size()
        _, seq_len_B, _ = drug.size()

        query = self.query(cell)
        key   = self.key(drug)
        value = self.value(drug)

        query = query.view(batch_size, seq_len_A, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key   = key.view(batch_size, seq_len_B, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len_B, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        if output_attention:
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if cell_attention_mask is not None:
                attention_scores = attention_scores + cell_attention_mask
            attention_probs = F.softmax(attention_scores, dim=-1)
            context = torch.matmul(attention_probs, value)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_A, hidden_size)
            return context, attention_probs
        else:
            context = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0
            )
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_A, hidden_size)
            return context, None


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SublayerConnection, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        return hidden_states + input_tensor


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.feed_forward(input_tensor)
        hidden_states = self.sublayer(hidden_states, input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)
        self.output = SelfOutput(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None, output_attention=False):
        attention_output, attention_probs = self.attention(input_tensor, attention_mask, output_attention)
        attention_output = self.sublayer(attention_output, input_tensor)
        layer_output = self.output(attention_output)
        return layer_output, attention_probs


class crossEncoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(crossEncoder, self).__init__()
        self.attention_CA = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.attention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.sublayer = SublayerConnection(hidden_size, hidden_dropout_prob)
        self.output = SelfOutput(hidden_size, intermediate_size, hidden_dropout_prob)
        self.drug_SA = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)

    def forward(self, cell, drug, drug_attention_mask=None, cell_attention_mask=None, output_attention=False):
        drug_SA_embed, _ = self.drug_SA(drug, drug_attention_mask, output_attention)

        cell_attention_out_0, cell_attention_probs_0 = self.attention(cell, cell_attention_mask, output_attention)
        cell_embed = self.sublayer(cell_attention_out_0, cell)

        cell_attention_output_1, cell_attention_probs_1 = self.attention_CA(cell_embed, drug_SA_embed, cell_attention_mask, drug_attention_mask, output_attention)
        cell_embed = self.sublayer(cell_attention_output_1, cell_embed)
        cell_intermediate_output = self.output(cell_embed)

        return cell_intermediate_output, drug_SA_embed, cell_attention_probs_1

class AttnEncoder(nn.Module):
    def __init__(self, config, structure):
        super(AttnEncoder, self).__init__()

        hidden_size = config['model']['ATTN']['hidden_size']
        intermediate_size = hidden_size * 2
        num_attention_heads = config['model']['ATTN']['n_heads']
        attention_probs_dropout_prob = config['model']['ATTN']['attention_probs_dropout_prob']
        hidden_dropout_prob = config['model']['ATTN']['hidden_dropout_prob']

        self.structure = structure
        self.layers = self.structure.split('+')
        CA_N = self.layers.count('CA')
        SA_N = self.layers.count('SA')

        self.crossEncoders = nn.ModuleList(
            [crossEncoder(hidden_size, intermediate_size, num_attention_heads,
                          attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(CA_N)]
        )
        self.selfEncoders = nn.ModuleList(
            [Encoder(hidden_size, intermediate_size, num_attention_heads,
                     attention_probs_dropout_prob, hidden_dropout_prob) for _ in range(SA_N)]
        )

    def forward(self, cell_embed, drug_embed=None, cell_attention_mask=None, drug_attention_mask=None, output_attention=False):
        CA_layer_idx = 0
        SA_layer_idx = 0
        attention_dict = {}

        for layer_count, layer_type in enumerate(self.layers):
            if layer_type == 'CA':
                cell_embed, drug_embed, attention = self.crossEncoders[CA_layer_idx](
                    cell_embed, drug_embed, drug_attention_mask, cell_attention_mask, output_attention)
                if output_attention:
                    attention_dict[f'CA_{layer_count}'] = attention
                CA_layer_idx += 1

            elif layer_type == 'SA':
                cell_embed, attention = self.selfEncoders[SA_layer_idx](
                    cell_embed, cell_attention_mask, output_attention)
                if output_attention:
                    attention_dict[f'SA_{layer_count}'] = attention
                SA_layer_idx += 1

        if output_attention:
            return cell_embed, drug_embed, attention_dict
        return cell_embed, drug_embed, None


class XPertMorgan(torch.nn.Module):
    def __init__(self, config, device):
        super(XPertMorgan, self).__init__()

        self.device = device

        max_gene_length = config['dataset']['gene_num']
        exp_vocab_size = config['dataset']['n_bins']

        hidden_size = config['model']['ATTN']['hidden_size']
        self.hidden_size = hidden_size
        cell_input_hidden_dropout_prob = config['model']['ATTN']['cell_input_hidden_dropout_prob']
        drug_input_hidden_dropout_prob = config['model']['ATTN']['drug_input_hidden_dropout_prob']

        pretrained_ppi_embed_path = config['model']['ATTN']['ppi_gene_vector_path']
        self.cell_emb = cell_Embeddings(
            exp_vocab_size, hidden_size, max_gene_length,
            cell_input_hidden_dropout_prob, pretrained_ppi_embed_path
        )

        pert_time_emb = nn.Embedding(config['dataset']['num_pert_time'], hidden_size)
        self.drug_emb = morgan_Embeddings(
        hidden_size, drug_input_hidden_dropout_prob, pert_time_emb)

        ctl_structure = config['model']['ATTN']['ctl_structure']
        trt_structure = config['model']['ATTN']['trt_structure']
        self.attnEncoder_ctl = AttnEncoder(config, ctl_structure)
        self.attnEncoder_trt = AttnEncoder(config, trt_structure)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        num_cell_id = config['dataset']['num_cell_id']
        self.class_fc = nn.Linear(hidden_size, num_cell_id)

        latent_size = 64
        self.ctl_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )
        self.trt_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )
        self.deg_fc = nn.Sequential(
            nn.Linear(hidden_size, latent_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(latent_size, 1),
        )

    def forward(self, data, output_attention=False):
        trt_raw_data, ctl_raw_data, ctl_raw_data_binned, drug_feat, pert_time_idx, cell_idx = data

        trt_raw_data = trt_raw_data.to(self.device)
        ctl_raw_data = ctl_raw_data.to(self.device)
        ctl_raw_data_binned = ctl_raw_data_binned.to(self.device)
        drug_feat = drug_feat.to(self.device)
        pert_time_idx = pert_time_idx.to(self.device)
        cell_idx = cell_idx.to(self.device)

        cell_class_true = cell_idx
        num_samples = cell_idx.shape[0]

        drug_embed = self.drug_emb(drug_feat, pert_time_idx)
        cell_embed = self.cell_emb(ctl_raw_data_binned)

        cls_tokens = self.cls_token.expand(num_samples, -1, -1)
        cell_embed = torch.cat([cls_tokens, cell_embed], dim=1)

        trt_cell_embed_attn, drug_embed, trt_attention_dict = self.attnEncoder_trt(
            cell_embed, drug_embed, None, None, output_attention)
        ctl_cell_embed_attn, _, ctl_attention_dict = self.attnEncoder_ctl(
            cell_embed, None, None, None, output_attention)

        trt_cell_embed = trt_cell_embed_attn[:, 1:, :]
        ctl_cell_embed = ctl_cell_embed_attn[:, 1:, :]
        trt_cls_embed = trt_cell_embed_attn[:, 0, :]
        ctl_cls_embed = ctl_cell_embed_attn[:, 0, :]

        cell_class_predict_1 = self.class_fc(ctl_cls_embed)
        cell_class_predict_2 = self.class_fc(trt_cls_embed)
        cell_class_predict = (cell_class_predict_1, cell_class_predict_2)

        trt_output = self.trt_fc(trt_cell_embed).squeeze(-1)
        ctl_output = self.ctl_fc(ctl_cell_embed).squeeze(-1)
        deg_output = self.deg_fc(trt_cell_embed - ctl_cell_embed).squeeze(-1)

        attention_dict = (trt_attention_dict, ctl_attention_dict)

        return trt_output, ctl_output, deg_output, trt_raw_data, ctl_raw_data, \
        attention_dict, cell_class_true, cell_class_predict

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def compute_bin_edges(train_expr, n_bins):
    flat = train_expr.flatten()
    percentiles = np.linspace(0, 100, n_bins + 1)
    return np.percentile(flat, percentiles)


def apply_binning(expr, bin_edges):
    n_bins = len(bin_edges) - 1
    binned = np.digitize(expr, bin_edges[1:-1])
    return np.clip(binned, 0, n_bins - 1).astype(np.int64)

def precision_at_k(targets, preds, k=20, direction='pos'):
    if direction == 'pos':
        true_top = set(np.argsort(targets)[-k:])
        pred_top = set(np.argsort(preds)[-k:])
    else:
        true_top = set(np.argsort(targets)[:k])
        pred_top = set(np.argsort(preds)[:k])
    return len(true_top & pred_top) / k

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch, output_attention=False)
            trt_output, ctl_output, deg_output, trt_raw, ctl_raw, \
                attn, cell_class_true, cell_class_predict = outputs
            all_preds.append(deg_output.cpu().numpy())
            all_targets.append(trt_raw.cpu().numpy())

    all_preds   = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    n = len(all_targets)

    pccs = [pearsonr(all_targets[i], all_preds[i])[0] for i in range(n)]
    spearmans = [spearmanr(all_targets[i], all_preds[i])[0] for i in range(n)]
    r2s = [r2_score(all_targets[i], all_preds[i]) for i in range(n)]
    rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    pos_p20 = [precision_at_k(all_targets[i], all_preds[i], 20, 'pos') for i in range(n)]
    neg_p20 = [precision_at_k(all_targets[i], all_preds[i], 20, 'neg') for i in range(n)]

    print(f'Mean PCC: {np.mean(pccs):.4f} ± {np.std(pccs):.4f}')
    print(f'Median PCC: {np.median(pccs):.4f}')
    print(f'Mean Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}')
    print(f'Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'Mean Most Upregulated 20 Genes: {np.mean(pos_p20):.4f} ± {np.std(pos_p20):.4f}')
    print(f'Mean Most Downregulated 20 Genes: {np.mean(neg_p20):.4f} ± {np.std(neg_p20):.4f}')

    return {
        'pccs':      np.array(pccs),
        'spearmans': np.array(spearmans),
        'r2s':       np.array(r2s),
        'rmse':      rmse,  # scalar
        'pos_p20':   np.array(pos_p20),
        'neg_p20':   np.array(neg_p20),
    }


def loss_fn(deg_output, trt_raw, cell_class_predict, cell_class_true,
            mse_weight, pcc_weight):
    mse = F.mse_loss(deg_output, trt_raw)

    # differentiable 1 - mean PCC across samples in the batch
    deg_c = deg_output - deg_output.mean(dim=1, keepdim=True)
    trt_c = trt_raw   - trt_raw.mean(dim=1, keepdim=True)
    pcc = (deg_c * trt_c).sum(dim=1) / (
        deg_c.norm(dim=1) * trt_c.norm(dim=1) + 1e-8
    )
    pcc_loss = 1.0 - pcc.mean()

    ctl_logits, trt_logits = cell_class_predict
    ce = (F.cross_entropy(ctl_logits, cell_class_true) +
          F.cross_entropy(trt_logits, cell_class_true)) / 2.0
    cls_weight = 0.003
    return mse_weight * mse + pcc_weight * pcc_loss + cls_weight * ce