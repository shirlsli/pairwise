# Sequential Drug Combination Project Report
## Investigating LINCS workshop Data
I first tried looking at the data obtained from this [workshop Github](https://github.com/cmap/lincs-workshop-2020/blob/main/notebooks/cell_fitness/Exercise_1_Exploration_of_Prism_Pr500_Cell_Viability_data.ipynb).

I downloaded the data and looked at it manually. 

```bash
 curl -s https://s3.amazonaws.com/repo-assets.clue.io/lincswkshp20_pfc_assets.tgz | tar zx -C .
```

I couldn't find any data that seemed to be related to cell viability, so I will be using the CTRPv2 dataset instead.

## Pre-processing LINCS L1000, CTRPv2, and Drug Smiles
### LINCS L1000
Since I need to predict change of expression, I need to pair the post-treatment vs. control LINCS data together as input data. As such, I can't use the level 5 data which gives the z-scores. Instead I'm using the level 3 data similar to the paper.

I considered using the PRISM Repurposing dataset since it seems to be newer compared to CTRPv2. 