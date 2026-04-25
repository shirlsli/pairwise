from preprocess_p13 import preprocess_p13
import matplotlib.pyplot as plt

def pipeline_p13():
    p13_df = preprocess_p13(
        input_path='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/synergy_data/p13/p13_trueset.csv',
        output_path='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/synergy_data/p13/p13_preprocessed.csv',
        lincs_pkl_path='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/perturbation_data/processed_lincs_6_24_hrs_9-11_uM_cell_baseline_consensus.pkl'
    )
    print(f"Unique drug1: {p13_df['drug_name_x'].nunique()}")
    print(f"Unique drug2: {p13_df['drug_name_y'].nunique()}")
    print(f"Unique cells: {p13_df['DepMap_ID'].nunique()}")
    p13_df['synergy_bliss'].hist(bins=50)
    plt.title("Bliss Synergy Score Distribution")
    plt.savefig("bliss_score_dist.png")

    print(p13_df['synergy_bliss'].describe())
    return p13_df

if __name__ == "__main__":
    pipeline_p13()