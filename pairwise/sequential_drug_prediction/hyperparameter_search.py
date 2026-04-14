import torch
import optuna
from torch.utils.data import DataLoader
from xpert_morgan import XPertMorgan, PerturbationDataset, evaluate_model, loss_fn
import numpy as np
import os
import pickle
import json

def objective(trial, train_data, val_data, cell_id_to_idx, device, base_config):
    hidden_size = 256
    batch_size = 256
    num_heads = 8
    learning_rate = trial.suggest_float('learning_rate', 5.6e-3, 8e-3, log=True)
    ctl_structure = 'SA+SA+SA+SA'
    trt_structure = trial.suggest_categorical('trt_structure', ['CA+SA+SA+CA', 'CA+SA+SA+SA+CA'])
    mse_weight = trial.suggest_categorical('mse_weight', [0.1, 0.2, 0.5])
    pcc_weight = trial.suggest_categorical('pcc_weight', [0.5, 1.0, 2.0])
    dropout = trial.suggest_float('dropout', 0.0, 0.1)

    config = {
        'dataset': base_config['dataset'].copy(),
        'model': {
            'ATTN': {
                'hidden_size': hidden_size,
                'n_heads': num_heads,
                'attention_probs_dropout_prob': dropout,
                'hidden_dropout_prob': dropout,
                'cell_input_hidden_dropout_prob': dropout,
                'drug_input_hidden_dropout_prob': dropout,
                'ppi_gene_vector_path': base_config['model']['ATTN']['ppi_gene_vector_path'],
                'ctl_structure': ctl_structure,
                'trt_structure': trt_structure,
            }
        }
    }
    config['dataset']['num_cell_id'] = len(cell_id_to_idx)

    train_dataset = PerturbationDataset(train_data, cell_id_to_idx)
    val_dataset   = PerturbationDataset(val_data,   cell_id_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    valid_loader = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = XPertMorgan(config, device)
    model.init_weights()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_val_loss    = float('inf')
    patience_counter = 0
    model_path       = f'optuna_trial_{trial.number}.pt'

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            trt_output, ctl_output, deg_output, trt_raw, ctl_raw, \
                attn, cell_class_true, cell_class_predict = outputs
            loss = loss_fn(deg_output, trt_raw, cell_class_predict,
                           cell_class_true, mse_weight, pcc_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                outputs = model(batch)
                trt_output, ctl_output, deg_output, trt_raw, ctl_raw, \
                    attn, cell_class_true, cell_class_predict = outputs
                val_loss += loss_fn(deg_output, trt_raw, cell_class_predict,
                                    cell_class_true, mse_weight, pcc_weight).item()
        val_loss /= len(valid_loader)

        print(f"Epoch {epoch + 1}/100 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.exceptions.TrialPruned()

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    metrics = evaluate_model(model, valid_loader, device)

    if os.path.exists(model_path):
        os.remove(model_path)

    return -float(np.mean(metrics['pccs']))


def run_hyperparameter_search(all_splits, device, n_trials=20,
                               output_path='optuna_study.pkl', base_config=None):
    split = all_splits[0]
    train_data = split['train']
    val_data = split['val']

    all_cell_ids = np.unique(np.concatenate([
        split['train']['cell_ids'],
        split['val']['cell_ids'],
        split['test']['cell_ids']
    ]))
    cell_id_to_idx = {c: i for i, c in enumerate(all_cell_ids)}

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    study.optimize(
        lambda trial: objective(
            trial, train_data, val_data, cell_id_to_idx, device, base_config),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\nCompleted trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Failed trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    print(f"\nBest trial:")
    print(f"  PCC: {-study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    with open(output_path, 'wb') as f:
        pickle.dump(study, f)

    return study.best_trial.params