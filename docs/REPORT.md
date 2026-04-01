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

I wasn't totally sure which parameters to choose, so I had Claude generate hyperparameter choices.

**hidden_size**: [64, 128, 256]
**num_heads**: [4, 8]
**ctl_structure**: ['SA', 'SA+SA', 'SA+SA+SA'] (number of self-attention layers)
**trt_structure**: ['CA+SA', 'CA+SA+CA', 'CA+SA+CA+SA'] (layer structure, cross attention = CA, SA = self-attention, XPert's default is CA+SA+CA.)
**learning_rate**: [1e-4, 1e-2]
**mse_weight**: [0.5, 2.0]
**pcc_weight**: [0.5, 5.0]
**drug_hidden_dim**: [256, 512, 1024]
**dropout**: [0.1, 0.3]

I've submitted a bash script `model_comparison.sh` to SLURM and am waiting for it to be allocated a node. Running on CPU may take a long time.

#### Citations

Chen, X., Deng, Y., Yang, X. et al. Reinforcement learning-based design of sequential drug treatment targeting the evolving tumour landscape with SequenTx. Nat Mach Intell 8, 351–371 (2026). https://doi.org/10.1038/s42256-026-01192-1

Clark, N.R., Hu, K.S., Feldmann, A.S. et al. The characteristic direction: a geometrical approach to identify differentially expressed genes. BMC Bioinformatics 15, 79 (2014). https://doi.org/10.1186/1471-2105-15-79

Duan, Q., Reid, S., Clark, N. et al. L1000CDS2: LINCS L1000 characteristic direction signatures search engine. npj Syst Biol Appl 2, 16015 (2016). https://doi.org/10.1038/npjsba.2016.15

Guo, Y., Zhang, H., Hu, H. et al. Modelling drug-induced cellular perturbation responses with a biologically informed dual-branch transformer. Nat Mach Intell 8, 96–112 (2026). https://doi.org/10.1038/s42256-025-01165-w

Tong, X., Qu, N., Kong, X. et al. Deep representation learning of chemical-induced transcriptional profile for phenotype-based drug discovery. Nat Commun 15, 5378 (2024). https://doi.org/10.1038/s41467-024-49620-3