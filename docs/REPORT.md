# Sequential Drug Combination Project Progress Report (Perturbation Model Part)
## Investigating LINCS workshop Data
I first tried looking at the data obtained from this [workshop Github](https://github.com/cmap/lincs-workshop-2020/blob/main/notebooks/cell_fitness/Exercise_1_Exploration_of_Prism_Pr500_Cell_Viability_data.ipynb).

I downloaded the data and looked at it manually. 

```bash
 curl -s https://s3.amazonaws.com/repo-assets.clue.io/lincswkshp20_pfc_assets.tgz | tar zx -C .
```

I couldn't find any data that seemed to be related to cell viability, so I will be using the CTRPv2 dataset instead.

## Pre-processing LINCS L1000 and Drug Smiles
### LINCS L1000
Since I need to predict change of expression, I need to pair the post-treatment vs. control LINCS data together as input data, so I'm using the level 3 data similar to SequenTX.

I decided to include dose duration as an additional input. I filtered the drug dosage to be between 9-11 µM similar to the SequenTX paper and chose the top most common durations based on those dosages for the drugs I was interested in, which were 6 hrs and 24 hrs.

When condensing the biological replicates, I was trying to apply characteristic direction (CD) in addition to MODZ (applied by TranSiGen) due to Duan et. al's conclusion that CD can capture a better noise-to-signal ratio compared to MODZ for LINCS data. I tried out the [package created by Clark et. al](http://www.maayanlab.net/CD/), but I wasn't able to get CDs for every compound so decided to stick with only MODZ (the CD package threw errors for a sizable amount of compounds), which is used by TranSiGen during preprocessing.

```
Number of unique cell lines: 70
Unique cell lines:
['A375' 'A549' 'A673' 'AGS' 'ASC' 'BT20' 'CD34' 'CL34' 'CORL23' 'COV644'
 'DV90' 'EFO27' 'FIBRNPC' 'H1299' 'HA1E' 'HCC15' 'HCC515' 'HCT116'
 'HEC108' 'HEK293T' 'HEPG2' 'HL60' 'HS27A' 'HS578T' 'HT115' 'HT29' 'HUH7'
 'JHUEM2' 'JURKAT' 'LOVO' 'MCF10A' 'MCF7' 'MDAMB231' 'MDST8' 'NCIH1694'
 'NCIH1836' 'NCIH2073' 'NCIH508' 'NCIH596' 'NEU' 'NKDBA' 'NOMO1' 'NPC'
 'OV7' 'PC3' 'PHH' 'PL21' 'RKO' 'RMGI' 'RMUGS' 'SKB' 'SKBR3' 'SKLU1'
 'SKM1' 'SKMEL1' 'SKMEL28' 'SNGM' 'SNU1040' 'SNUC4' 'SNUC5' 'SW480'
 'SW620' 'SW948' 'T3M10' 'THP1' 'TYKNU' 'U266' 'U937' 'VCAP' 'WSUDLCL2']
```

### Overall Preprocessing Steps for Perturbation Model (separated into `preprocess_lincs.py`, `preprocess_drugs.py`, and `train_val_test_split.py`):
- Compute Morgan fingerprint from LINCS pert info file for each compound
- Extract expression profiles for the 978 landmark genes in LINCS L1000 filtered by 9-11 µM and 6 or 24 hrs
- Match each perturbation instance with DMSO control instance in same plate
- Apply MODZ to cell line baselines (DMSO replicates) across entire dataset instead of just original plate to collapse into single consensus profile
    - Might give more info/provide more accurate estimate about unperturbed cell state because includes all DMSO replicates for that cell line
    - Done when called with `--cell_line_consensus`
- Apply MODZ to treatment replicates to collapse into single consensus treatment profile
- Create time bin indices for dose durations (same bins used in XPert)
    - Including concentration unecessary for this particular preprocessing since 9-11 µM would fall into the same bin, leading to the concentration input to be constant
- Create quantile-based edges for expression profiles (applying global bin edges similar to XPert but instead of using uniform edges, quantile edges are used since MODZ scores are not necessarily uniformly distributed)
- Drugs randomly shuffled and divided into 10 folds for k-fold splitting
    - For each test fold, remaining 9 folds form train and validation pool, from which 1/10 of compounds are used for validation
    - Splits are compound-stratified (drug only appears once in either train, val, or test)
    - Split = (Morgan drug fingerprint, cell line baseline (raw), cell line baseline (binned), MODZ delta expression (target), time bin idx, cell ids, and drug names)

### Model Input

```
Input = (
    Morgan drug fingerprint,  # (1024,) float32
    cell line baseline binned, # (978,)  int64
    time bin idx,              # scalar  int64
    delta expression target    # (978,)  float32
)
```

## Transformer Encoder-Decoder with Cross-Attention

Guo et. al uses a transformer based implementation called XPert for predicting gene expression changes from LINCS 2020 data. It mentioned there is "an unreported limitation of VAE-based models: a lack of robustness in blind tests relative to attention-based approaches. or example, the leading VAE model, TranSiGen, performed well in warm-start tests but its performance deteriorated in cold-cell settings, scoring negative R2 values despite good correlation (Fig. 2b), suggesting a failure to adapt to unseen cellular contexts. We attribute this failure to two intrinsic VAE properties. First, the Kullback–Leibler divergence regularizer forces information compression that can lead to over-denoising, erasing critical cellular context features needed for gene-specific reconstruction. A typical example is the generation of blurry images in image generation by VAEs. Second, VAEs are constrained by their training data, leading to low-fidelity outputs when encountering out-of-distribution samples like unseen cell lines."

This led me to consider incorporating a simplified version of the dual-branch transformer design from XPert. XPert has two outputs: perturbed profile and gene expression changes. Since I'm only interested in gene expression changes, I will only have 1 output head (vs. the 3 in XPert) and am using the same gene expression delta loss function as XPert. According to Guo et. al, "the loss for this task is a combination of m.s.e. and PCC losses. By incorporating the PCC loss, the model is encouraged to not only minimize the absolute differences between predictions and ground truth but also to capture the underlying correlation structure, leading to more accurate and biologically meaningful predictions".

`Ldeg = β * MSE(xdeg, x̂deg) + γ * (1 - PCC(xdeg, x̂deg))`

**β and γ are weighting coefficients**

XPert credits the pretrained heterogenous graph biological embedding (includes information about drug-target interactions, protein-protein interactions, and drug-drug structural similarity) for its cold-start performance, so I decided to incorporate that as well. I submitted an Academic Downloads license application to DrugBank and am currently waiting on their response. For now, I will not include this, but I plan on retraining once I receive access. XPert also uses UniMol to represent compounds, but I chose Morgan fingerprints for simplicity's sake.

### Model Architecture (Guo et. al)

While XPert could serve as a starting point, I adapted it as a 
transition function for SequenTx-style sequential drug combination 
prediction. The key modifications are: (1) single output head 
predicting only delta expression (xdeg) with Ldeg loss only, removing 
the absolute expression and cell-type classification heads; (2) Morgan 
fingerprint drug representation replacing XPert's UniMol 3D features; and 
(3) MODZ-based preprocessing using cell-line-wide DMSO consensus 
baselines and quantile-based expression binning.

#### Baseline Encoder Branch

"The base encoder captures the unperturbed state of the cell by learning the dependencies between genes within the cell. It utilizes stacked self-attention layers to iteratively process the initial gene expression representation of the unperturbed cell. Given the initial representation, the encoder sequentially applies self-attention blocks across n layers."

#### Perturbation Encoder Branch

"The Pert encoder is responsible for integrating drug molecular features with cellular context through cascaded cross-attention and self-attention layers. The cross-attention module explicitly models gene-level perturbation effects by aligning the multimodal drug representation with cellular-state features. Subsequent self-attention layers refine these interaction patterns and maintain the positional awareness of key regulatory genes.

In the cross-attention layers, the cell representation is treated as the query, and tokenized drug representation serves as the key and value matrix. This allows the model to learn gene-level perturbation effects induced by the drug."

#### Hyperparameter Search

##### Summary of Manual Refinement Decisions

The hyperparameter tuning proceeded in two stages: iterative manual refinement of the training loop followed by automated Bayesian search using Optuna.

The initial training run (sequential fold training, no learning rate scheduler, 500 epochs, patience 50) was cancelled due to Cayuga's job time limit. To speed up the training process, the following measures were implemented:
- FlashAttention via using `scaled_dot_product_attention` in self-attention and cross-attention layers
- Decreased from 10-fold to 5-fold cross-validation
    - Helped with parallelizing fold training since it took up less nodes than 10-fold
- Shortened patience from 50 to 20
    - Initially used 500 epochs similar to paper, but decreased to 200 epochs to stay within the time limit
- Guo et al mentioned for larger datasets would be better to consider larger batch sizes and more attention layers
    - Doubled batch size from 128 to 256
    - Increased learning rate from `4e-3` to `5.6e-3` (chose to experiment between multiplying by sqrt(batch size multiplication factor) and linearly scaling)
    - Also tried training with linearly scaled learning rate `8e-3`
        - Currently still running
- Added learning rate scheduler
    - Linear warmup over 70 epochs (LR increases from `base_lr / warmup_epochs` to `base_lr`) followed by cosine annealing back to zero over the remaining 130 epochs. This schedule was chosen because early epochs with high LR caused erratic validation loss (train loss spiking to >4.0 in trial 0 of the Optuna search at the same LR range), while the gradual warmup allowed stable convergence. The `base_lr` was set to `4e-3`, peaking at approximately `5.6e-3` at epoch 70
    - I realized I messed up after running the base config model because I didn't checkpoint the optimizer and learning rate scheduler states, so that function was added in the CLI as well
- AdamW with `weight_decay=1e-5` and gradient clipping at norm 1.0 used for optimizer
- Using `mse_weight=0.2`, `pcc_weight=1.0` as loss weights similar to Guo et al

##### Optuna Bayesian search

An Optuna study (TPE sampler, seed 42; MedianPruner with 5 warmup steps) was submitted to search over the following variables while holding the fixed parameters constant:

| Parameter | Search range |
|---|---|
| `learning_rate` | Log-uniform in [5.6×10⁻³, 8×10⁻³] |
| `trt_structure` | {`CA+SA+SA+CA`, `CA+SA+SA+SA+CA`} |
| `mse_weight` | {0.1, 0.2, 0.5} |
| `pcc_weight` | {0.5, 1.0, 2.0} |
| `dropout` | Uniform in [0.0, 0.1] |

The search was configured for 20 trials, each trained for up to 100 epochs with patience 10 on fold 0's validation set and evaluated by mean PCC. Only 2 of the 20 planned trials have been completed so far:

| Trial | `learning_rate` | `trt_structure` | `mse_weight` | `pcc_weight` | `dropout` | Val PCC |
|---|---|---|---|---|---|---|
| 0 | 6.40×10⁻³ | `CA+SA+SA+CA` | 0.1 | 1.0 | 0.071 | 0.6597 |
| 1 | 5.64×10⁻³ | `CA+SA+SA+CA` | 0.1 | 1.0 | 0.029 | 0.6889 |

Trial 1 was the best found, suggesting that lower dropout (0.029 vs. 0.071) and a slightly lower peak learning rate improve validation PCC. Both trials preferred `mse_weight=0.1` over the manually chosen 0.2, and both retained the default `CA+SA+SA+CA` perturbation encoder structure. The search did not complete enough trials to draw firm conclusions about `pcc_weight` or `trt_structure`, so the base configuration was used for the full 5-fold evaluation.

### Results For Base Config

| Category | Parameter | Value |
|---|---|---|
| Dataset | `gene_num` | 978 |
| Dataset | `n_bins` | 128 |
| Dataset | `num_cell_id` | 70 |
| Dataset | `num_pert_time` | 6 |
| Model | `hidden_size` | 256 |
| Model | `n_heads` | 8 |
| Model | `attention_probs_dropout_prob` | 0.1 |
| Model | `hidden_dropout_prob` | 0.1 |
| Model | `cell_input_hidden_dropout_prob` | 0.1 |
| Model | `drug_input_hidden_dropout_prob` | 0.1 |
| Model | `ctl_structure` | `SA+SA+SA+SA` |
| Model | `trt_structure` | `CA+SA+SA+CA` |
| Training | `learning_rate` | 4×10⁻³ |
| Training | `mse_weight` | 0.2 |
| Training | `pcc_weight` | 1.0 |
| Training | `batch_size` | 128 |
| Training | `epochs` | 200 |
| Training | `patience` | 50 |
| Training | `warmup_epochs` | 70 |
| Training | `weight_decay` | 1×10⁻⁵ |
| Training | `grad_clip_norm` | 1.0 |

All five folds trained to completion (200 epochs) with no early stopping triggered on these parameters. Validation loss decreased relatively smoothly across the folds.

![Training loss curves and test metrics across all 5 folds](../pairwise/sequential_drug_prediction/training_summary_2757168.png)

**Table 1. Per-fold test metrics (compound-stratified 5-fold cross-validation).**

| Fold | PCC (mean ± std) | Spearman (mean ± std) | R² (mean ± std) | RMSE | Prec@20 Up | Prec@20 Down |
|---|---|---|---|---|---|---|
| 0 | 0.6977 ± 0.1932 | 0.6850 ± 0.1938 | 0.4624 ± 0.3277 | 0.5348 | 0.4377 ± 0.1863 | 0.3754 ± 0.1704 |
| 1 | 0.7041 ± 0.1899 | 0.6920 ± 0.1899 | 0.4718 ± 0.3245 | 0.5305 | 0.4414 ± 0.1841 | 0.3817 ± 0.1708 |
| 2 | 0.7006 ± 0.1902 | 0.6885 ± 0.1905 | 0.4642 ± 0.3450 | 0.5294 | 0.4382 ± 0.1846 | 0.3771 ± 0.1712 |
| 3 | 0.7085 ± 0.1894 | 0.6962 ± 0.1900 | 0.4800 ± 0.3222 | 0.5259 | 0.4492 ± 0.1862 | 0.3882 ± 0.1724 |
| 4 | 0.6955 ± 0.1947 | 0.6829 ± 0.1952 | 0.4604 ± 0.3355 | 0.5345 | 0.4326 ± 0.1859 | 0.3741 ± 0.1710 |
| **Mean** | **0.7013 ± 0.0046** | **0.6889 ± 0.0048** | **0.4678 ± 0.0072** | **0.5310 ± 0.0033** | **0.4398 ± 0.0055** | **0.3793 ± 0.0051** |

## Citations

Chen, X., Deng, Y., Yang, X. et al. Reinforcement learning-based design of sequential drug treatment targeting the evolving tumour landscape with SequenTx. Nat Mach Intell 8, 351–371 (2026). https://doi.org/10.1038/s42256-026-01192-1

Clark, N.R., Hu, K.S., Feldmann, A.S. et al. The characteristic direction: a geometrical approach to identify differentially expressed genes. BMC Bioinformatics 15, 79 (2014). https://doi.org/10.1186/1471-2105-15-79

Duan, Q., Reid, S., Clark, N. et al. L1000CDS2: LINCS L1000 characteristic direction signatures search engine. npj Syst Biol Appl 2, 16015 (2016). https://doi.org/10.1038/npjsba.2016.15

Guo, Y., Zhang, H., Hu, H. et al. Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer. Nat Mach Intell 8, 96–112 (2026). https://doi.org/10.1038/s42256-025-01165-w

Tong, X., Qu, N., Kong, X. et al. Deep representation learning of chemical-induced transcriptional profile for phenotype-based drug discovery. Nat Commun 15, 5378 (2024). https://doi.org/10.1038/s41467-024-49620-3