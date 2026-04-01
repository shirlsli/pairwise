import optuna
from torch.utils.data import DataLoader
from xpert_morgan import XPertMorgan, PerturbationDataset, evaluate_model
import numpy as np

def objective(trial, train_data, val_data, device):
    """
    Optuna objective function. Called once per trial.
    Returns validation metric to minimize (negative PCC since
    Optuna minimizes by default).
    """
    # Define hyperparameter search space
    hidden_size    = trial.suggest_categorical('hidden_size', [64, 128, 256])
    num_heads      = trial.suggest_categorical('num_heads', [4, 8])
    ctl_structure  = trial.suggest_categorical('ctl_structure', ['SA', 'SA+SA', 'SA+SA+SA'])
    trt_structure  = trial.suggest_categorical('trt_structure', ['CA+SA', 'CA+SA+CA', 'CA+SA+CA+SA'])
    learning_rate  = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    mse_weight     = trial.suggest_float('mse_weight', 0.5, 2.0)
    pcc_weight     = trial.suggest_float('pcc_weight', 0.5, 5.0)
    drug_hidden_dim = trial.suggest_categorical('drug_hidden_dim', [256, 512, 1024])
    dropout        = trial.suggest_float('dropout', 0.0, 0.3)

    train_dataset = PerturbationDataset(
        fingerprints=train_data['drug_fp'],
        binned_expr=train_data['ccl_binned'],
        time_idx=train_data['time_idx'],
        y=train_data['delta_expr']
    )
    val_dataset = PerturbationDataset(
        fingerprints=val_data['drug_fp'],
        binned_expr=val_data['ccl_binned'],
        time_idx=val_data['time_idx'],
        y=val_data['delta_expr']
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    model = XPertMorgan(
        gene_number=train_data['delta_expr'].shape[1],
        hidden_size=hidden_size,
        num_heads=num_heads,
        ctl_structure=ctl_structure,
        trt_structure=trt_structure,
        learning_rate=learning_rate,
        mse_weight=mse_weight,
        pcc_weight=pcc_weight,
        drug_hidden_dim=drug_hidden_dim,
        attn_dropout=dropout,
        hidden_dropout=dropout,
        # Use fewer epochs during search to save time
        epoch=50,
        device=device,
        model_file=f'optuna_trial_{trial.number}.pt'
    )

    model.fit(train_loader, valid_loader)

    # Evaluate on validation set
    import torch
    model = torch.load(
        f'optuna_trial_{trial.number}.pt',
        map_location=device
    )
    pccs = evaluate_model(model, valid_loader)
    mean_pcc = float(np.mean(pccs))

    # Clean up trial checkpoint
    import os
    if os.path.exists(f'optuna_trial_{trial.number}.pt'):
        os.remove(f'optuna_trial_{trial.number}.pt')

    return -mean_pcc   # negative because Optuna minimizes


def run_hyperparameter_search(all_splits, device, n_trials=50,
                               output_path='optuna_study.pkl'):
    """
    Run Optuna search on fold 0 only — no need to search across
    all folds since the optimal hyperparameters are then fixed
    and used for the full 10-fold CV.
    """
    # Use fold 0 for hyperparameter search
    split = all_splits[0]
    train_data = split['train']
    val_data   = split['val']

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    study.optimize(
        lambda trial: objective(trial, train_data, val_data, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\nBest trial:")
    print(f"  Value (neg PCC): {study.best_trial.value:.4f}")
    print(f"  PCC:             {-study.best_trial.value:.4f}")
    print(f"  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # Save study for later analysis
    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump(study, f)

    return study.best_trial.params