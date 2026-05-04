import os
import pickle
import numpy as np
import argparse
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from xgboost import XGBRegressor


def generate_cell_viability_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    drug_fp = data['drug_fp']
    viability = data['viability']
    durations = data['durations']
    cell_ids = data['cell_ids']
    drug_names = data['drug_names']

    print(f'  Total samples:       {len(viability)}')
    print(f'  Delta expr shape:    {delta_expr.shape}')
    print(f'  Drug fp shape:       {drug_fp.shape}')
    print(f'  Unique cell lines:   {len(set(cell_ids))}')
    print(f'  Unique drugs:        {len(set(drug_names))}')
    print(f'  Timepoints present:  {sorted(set(durations))}')
    print(f'  Viability range:     [{viability.min():.3f}, {viability.max():.3f}]')

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}')
    print(f'Target vector Y shape:  {Y.shape}')

    np.random.seed(args.seed)
    sample_number = len(Y)
    print(f'\nSample size for training: {sample_number}')

    train_number = int(sample_number / 2)
    pcc = np.zeros(20)

    for i in range(20):
        test_ind = np.random.choice(sample_number, train_number, replace=False)

        all_ind = np.arange(sample_number)
        train_mask = np.ones(sample_number, dtype=bool)
        train_mask[test_ind] = False
        train_ind = all_ind[train_mask]

        train_x = X[train_ind]
        train_y = Y[train_ind]
        test_x = X[test_ind]
        test_y = Y[test_ind]

        model = Ridge(alpha=args.alpha)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)

        pcc[i] = np.corrcoef(test_y, pred_y)[0, 1]
        print(f'  Iteration {i+1:2d}/20  PCC: {pcc[i]:.4f}')

    print(f'\n20-fold Cross Validation PCC:')
    print(f'  Mean:   {pcc.mean():.4f}')
    print(f'  Std:    {pcc.std():.4f}')
    print(f'  Min:    {pcc.min():.4f}')
    print(f'  Max:    {pcc.max():.4f}')
    print(f'  Values: {pcc}')

    print('\nFitting final model on full dataset...')
    final_model = Ridge(alpha=args.alpha)
    final_model.fit(X, Y)

    model_bundle = {
        'model': final_model,
        'alpha': args.alpha,
        'delta_expr_dim': delta_expr.shape[1],
        'drug_fp_dim': drug_fp.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
        'pcc_cv': pcc,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(pcc, f)
    print(f'Saved PCC results to {result_out}')


def generate_gene_only_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    viability = data['viability']
    durations = data['durations']

    print(f'  Total samples:       {len(viability)}')
    print(f'  Delta expr shape:    {delta_expr.shape}')
    print(f'  Timepoints present:  {sorted(set(durations))}')
    print(f'  Viability range:     [{viability.min():.3f}, {viability.max():.3f}]')

    duration_col = durations.reshape(-1, 1).astype(np.float64)
    X = np.concatenate([delta_expr, duration_col], axis=1).astype(np.float64)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + duration)')
    print(f'Target vector Y shape:  {Y.shape}')

    np.random.seed(args.seed)
    sample_number = len(Y)
    train_number = int(sample_number / 2)
    print(f'\nSample size for training: {sample_number}')

    pcc = np.zeros(20)
    for i in range(20):
        test_ind = np.random.choice(sample_number, train_number, replace=False)
        train_mask = np.ones(sample_number, dtype=bool)
        train_mask[test_ind] = False
        train_ind = np.arange(sample_number)[train_mask]

        model = Ridge(alpha=args.alpha, solver='svd')
        model.fit(X[train_ind], Y[train_ind])
        pred_y = model.predict(X[test_ind])

        pcc[i] = np.corrcoef(Y[test_ind], pred_y)[0, 1]
        print(f'  Iteration {i+1:2d}/20  PCC: {pcc[i]:.4f}')

    print(f'\n20-fold Cross Validation PCC:')
    print(f'  Mean:   {pcc.mean():.4f}')
    print(f'  Std:    {pcc.std():.4f}')
    print(f'  Min:    {pcc.min():.4f}')
    print(f'  Max:    {pcc.max():.4f}')
    print(f'  Values: {pcc}')

    print('\nFitting final model on full dataset...')
    final_model = Ridge(alpha=args.alpha, solver='svd')
    final_model.fit(X, Y)

    model_bundle = {
        'model': final_model,
        'alpha': args.alpha,
        'delta_expr_dim': delta_expr.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
        'pcc_cv': pcc,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(pcc, f)
    print(f'Saved PCC results to {result_out}')


def generate_xgboost_gene_only_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    viability = data['viability']
    durations = data['durations']

    print(f'  Total samples:       {len(viability)}')
    print(f'  Delta expr shape:    {delta_expr.shape}')
    print(f'  Timepoints present:  {sorted(set(durations))}')
    print(f'  Viability range:     [{viability.min():.3f}, {viability.max():.3f}]')

    duration_col = durations.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([delta_expr, duration_col], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + duration)')
    print(f'Target vector Y shape:  {Y.shape}')

    np.random.seed(args.seed)
    sample_number = len(Y)
    train_number = int(sample_number / 2)
    print(f'\nSample size for training: {sample_number}')

    pcc = np.zeros(20)
    for i in range(20):
        test_ind = np.random.choice(sample_number, train_number, replace=False)
        train_mask = np.ones(sample_number, dtype=bool)
        train_mask[test_ind] = False
        train_ind = np.arange(sample_number)[train_mask]

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            tree_method='hist',
            device=args.device,
            random_state=args.seed,
            verbosity=0,
        )
        model.fit(X[train_ind], Y[train_ind])
        pred_y = model.predict(X[test_ind])

        pcc[i] = np.corrcoef(Y[test_ind], pred_y)[0, 1]
        print(f'  Iteration {i+1:2d}/20  PCC: {pcc[i]:.4f}')

    print(f'\n20-fold Cross Validation PCC:')
    print(f'  Mean:   {pcc.mean():.4f}')
    print(f'  Std:    {pcc.std():.4f}')
    print(f'  Min:    {pcc.min():.4f}')
    print(f'  Max:    {pcc.max():.4f}')
    print(f'  Values: {pcc}')

    print('\nFitting final model on full dataset...')
    final_model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method='hist',
        device=args.device,
        random_state=args.seed,
        verbosity=0,
    )
    final_model.fit(X, Y)

    model_bundle = {
        'model': final_model,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'delta_expr_dim': delta_expr.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
        'pcc_cv': pcc,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(pcc, f)
    print(f'Saved PCC results to {result_out}')


def generate_xgboost_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    drug_fp = data['drug_fp']
    viability = data['viability']

    print(f'  Total samples:       {len(viability)}')
    print(f'  Delta expr shape:    {delta_expr.shape}')
    print(f'  Drug fp shape:       {drug_fp.shape}')

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}')
    print(f'Target vector Y shape:  {Y.shape}')

    np.random.seed(args.seed)
    sample_number = len(Y)
    train_number = int(sample_number / 2)
    print(f'\nSample size for training: {sample_number}')

    pcc = np.zeros(20)
    for i in range(20):
        test_ind = np.random.choice(sample_number, train_number, replace=False)
        train_mask = np.ones(sample_number, dtype=bool)
        train_mask[test_ind] = False
        train_ind = np.arange(sample_number)[train_mask]

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            tree_method='hist',
            device=args.device,
            random_state=args.seed,
            verbosity=0,
        )
        model.fit(X[train_ind], Y[train_ind])
        pred_y = model.predict(X[test_ind])

        pcc[i] = np.corrcoef(Y[test_ind], pred_y)[0, 1]
        print(f'  Iteration {i+1:2d}/20  PCC: {pcc[i]:.4f}')

    print(f'\n20-fold Cross Validation PCC:')
    print(f'  Mean:   {pcc.mean():.4f}')
    print(f'  Std:    {pcc.std():.4f}')
    print(f'  Min:    {pcc.min():.4f}')
    print(f'  Max:    {pcc.max():.4f}')
    print(f'  Values: {pcc}')

    print('\nFitting final model on full dataset...')
    final_model = XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        tree_method='hist',
        device=args.device,
        random_state=args.seed,
        verbosity=0,
    )
    final_model.fit(X, Y)

    model_bundle = {
        'model': final_model,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'delta_expr_dim': delta_expr.shape[1],
        'drug_fp_dim': drug_fp.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
        'pcc_cv': pcc,
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(pcc, f)
    print(f'Saved PCC results to {result_out}')


def gridsearch_xgboost_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    drug_fp = data['drug_fp']
    viability = data['viability']

    print(f'  Total samples:    {len(viability)}')
    print(f'  Delta expr shape: {delta_expr.shape}')
    print(f'  Drug fp shape:    {drug_fp.shape}')

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.1, 0.3, 0.5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.7, 0.9],
    }

    pcc_scorer = make_scorer(lambda y, p: pearsonr(y, p)[0], greater_is_better=True)
    cv = KFold(n_splits=args.gs_folds, shuffle=True, random_state=args.seed)

    base = XGBRegressor(
        tree_method='hist',
        device=args.device,
        random_state=args.seed,
        verbosity=0,
    )

    n_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                      len(param_grid['learning_rate']) * len(param_grid['subsample']) *
                      len(param_grid['colsample_bytree']))
    print(f'\nRunning GridSearchCV: {n_combinations} combinations × {args.gs_folds} folds '
          f'= {n_combinations * args.gs_folds} fits  (n_jobs={args.n_jobs})')

    gs = GridSearchCV(base, param_grid, scoring=pcc_scorer, cv=cv,
                      n_jobs=args.n_jobs, refit=True, verbose=2)
    gs.fit(X, Y)

    print(f'\nBest PCC (CV): {gs.best_score_:.4f}')
    print(f'Best params:')
    for k, v in gs.best_params_.items():
        print(f'  {k}: {v}')

    model_bundle = {
        'model': gs.best_estimator_,
        'best_params': gs.best_params_,
        'best_score': gs.best_score_,
        'cv_results': gs.cv_results_,
        'delta_expr_dim': delta_expr.shape[1],
        'drug_fp_dim': drug_fp.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(gs.cv_results_, f)
    print(f'Saved GridSearch CV results to {result_out}')


def gridsearch_xgboost_gene_only_model(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    viability = data['viability']
    durations = data['durations']

    print(f'  Total samples:    {len(viability)}')
    print(f'  Delta expr shape: {delta_expr.shape}')
    print(f'  Timepoints present: {sorted(set(durations))}')

    duration_col = durations.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([delta_expr, duration_col], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + duration)')

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.1, 0.3, 0.5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.7, 0.9],
    }

    pcc_scorer = make_scorer(lambda y, p: pearsonr(y, p)[0], greater_is_better=True)
    cv = KFold(n_splits=args.gs_folds, shuffle=True, random_state=args.seed)

    base = XGBRegressor(
        tree_method='hist',
        device=args.device,
        random_state=args.seed,
        verbosity=0,
    )

    n_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                      len(param_grid['learning_rate']) * len(param_grid['subsample']) *
                      len(param_grid['colsample_bytree']))
    print(f'\nRunning GridSearchCV: {n_combinations} combinations × {args.gs_folds} folds '
          f'= {n_combinations * args.gs_folds} fits  (n_jobs={args.n_jobs})')

    gs = GridSearchCV(base, param_grid, scoring=pcc_scorer, cv=cv,
                      n_jobs=args.n_jobs, refit=True, verbose=2)
    gs.fit(X, Y)

    print(f'\nBest PCC (CV): {gs.best_score_:.4f}')
    print(f'Best params:')
    for k, v in gs.best_params_.items():
        print(f'  {k}: {v}')

    model_bundle = {
        'model': gs.best_estimator_,
        'best_params': gs.best_params_,
        'best_score': gs.best_score_,
        'cv_results': gs.cv_results_,
        'delta_expr_dim': delta_expr.shape[1],
        'feature_dim': X.shape[1],
        'landmark_gene_ids': data['landmark_gene_ids'],
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f'Saved model bundle to {model_out}')

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(gs.cv_results_, f)
    print(f'Saved GridSearch CV results to {result_out}')


def zero_shot_evaluation(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    drug_fp = data['drug_fp']
    viability = data['viability']
    cell_ids = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    print(f'  Total samples:       {len(viability)}')
    print(f'  Unique drugs:        {len(set(drug_names))}')
    print(f'  Unique cell lines:   {len(set(cell_ids))}')
    print(f'  Test fraction:       {args.test_frac}')
    print(f'  Iterations:          {args.zs_iters}')

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float64)
    Y = viability.astype(np.float32)

    unique_cell_lines = np.array(sorted(set(cell_ids)))

    np.random.seed(args.seed)
    pcc_scores = []
    iter_results = []

    for i in range(args.zs_iters):
        n_test_cells = max(1, int(len(unique_cell_lines) * args.test_frac))
        test_cells = set(np.random.choice(unique_cell_lines, n_test_cells, replace=False))
        train_cells = set(unique_cell_lines) - test_cells

        train_mask = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask = np.array([c in test_cells and d in train_drugs
                               for c, d in zip(cell_ids, drug_names)])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f'  Iteration {i+1:2d}/{args.zs_iters}  skipped (empty split)')
            continue

        model = Ridge(alpha=args.alpha, solver='svd')
        model.fit(X[train_mask], Y[train_mask])
        pred_y = model.predict(X[test_mask])

        pcc = np.corrcoef(Y[test_mask], pred_y)[0, 1]
        pcc_scores.append(pcc)
        iter_results.append({
            'iteration': i + 1,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'n_test_cells': len(test_cells),
            'pcc': float(pcc),
        })
        print(f'  Iteration {i+1:2d}/{args.zs_iters}  '
              f'train={train_mask.sum():5d}  test={test_mask.sum():5d}  '
              f'PCC: {pcc:.4f}  (held out {len(test_cells)} cell lines, drug seen)')

    pcc_arr = np.array(pcc_scores)
    print(f'\nZero-shot (cell line) PCC over {len(pcc_arr)} iterations:')
    print(f'  Mean:   {pcc_arr.mean():.4f}')
    print(f'  Std:    {pcc_arr.std():.4f}')
    print(f'  Min:    {pcc_arr.min():.4f}')
    print(f'  Max:    {pcc_arr.max():.4f}')

    results = {
        'test_frac': args.test_frac,
        'pcc_scores': pcc_arr,
        'iterations': iter_results,
    }

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved zero-shot results to {result_out}')


def zero_shot_xgboost_evaluation(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    drug_fp = data['drug_fp']
    viability = data['viability']
    cell_ids = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    print(f'  Total samples:       {len(viability)}')
    print(f'  Unique drugs:        {len(set(drug_names))}')
    print(f'  Unique cell lines:   {len(set(cell_ids))}')
    print(f'  Test fraction:       {args.test_frac}')
    print(f'  Iterations:          {args.zs_iters}')

    X = np.concatenate([delta_expr, drug_fp], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + drug_fp)')

    unique_cell_lines = np.array(sorted(set(cell_ids)))

    np.random.seed(args.seed)
    pcc_scores = []
    iter_results = []

    for i in range(args.zs_iters):
        n_test_cells = max(1, int(len(unique_cell_lines) * args.test_frac))
        test_cells = set(np.random.choice(unique_cell_lines, n_test_cells, replace=False))
        train_cells = set(unique_cell_lines) - test_cells

        train_mask = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask = np.array([c in test_cells and d in train_drugs
                               for c, d in zip(cell_ids, drug_names)])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f'  Iteration {i+1:2d}/{args.zs_iters}  skipped (empty split)')
            continue

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            tree_method='hist',
            device=args.device,
            random_state=args.seed + i,
            verbosity=0,
        )
        model.fit(X[train_mask], Y[train_mask])
        pred_y = model.predict(X[test_mask])

        pcc = np.corrcoef(Y[test_mask], pred_y)[0, 1]
        pcc_scores.append(pcc)
        iter_results.append({
            'iteration': i + 1,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'n_test_cells': len(test_cells),
            'pcc': float(pcc),
        })
        print(f'  Iteration {i+1:2d}/{args.zs_iters}  '
              f'train={train_mask.sum():5d}  test={test_mask.sum():5d}  '
              f'PCC: {pcc:.4f}  (held out {len(test_cells)} cell lines, drug seen)')

    pcc_arr = np.array(pcc_scores)
    print(f'\nZero-shot XGBoost (cell line) PCC over {len(pcc_arr)} iterations:')
    print(f'  Mean:   {pcc_arr.mean():.4f}')
    print(f'  Std:    {pcc_arr.std():.4f}')
    print(f'  Min:    {pcc_arr.min():.4f}')
    print(f'  Max:    {pcc_arr.max():.4f}')

    results = {
        'test_frac': args.test_frac,
        'pcc_scores': pcc_arr,
        'iterations': iter_results,
    }

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved zero-shot XGBoost results to {result_out}')


def zero_shot_gene_only_evaluation(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    viability = data['viability']
    durations = data['durations']
    cell_ids = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    print(f'  Total samples:       {len(viability)}')
    print(f'  Unique drugs:        {len(set(drug_names))}')
    print(f'  Unique cell lines:   {len(set(cell_ids))}')
    print(f'  Timepoints present:  {sorted(set(durations))}')
    print(f'  Test fraction:       {args.test_frac}')
    print(f'  Iterations:          {args.zs_iters}')

    duration_col = durations.reshape(-1, 1).astype(np.float64)
    X = np.concatenate([delta_expr, duration_col], axis=1).astype(np.float64)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + duration)')

    unique_cell_lines = np.array(sorted(set(cell_ids)))

    np.random.seed(args.seed)
    pcc_scores = []
    iter_results = []

    for i in range(args.zs_iters):
        n_test_cells = max(1, int(len(unique_cell_lines) * args.test_frac))
        test_cells = set(np.random.choice(unique_cell_lines, n_test_cells, replace=False))
        train_cells = set(unique_cell_lines) - test_cells

        train_mask = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask = np.array([c in test_cells and d in train_drugs
                               for c, d in zip(cell_ids, drug_names)])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f'  Iteration {i+1:2d}/{args.zs_iters}  skipped (empty split)')
            continue

        model = Ridge(alpha=args.alpha, solver='svd')
        model.fit(X[train_mask], Y[train_mask])
        pred_y = model.predict(X[test_mask])

        pcc = np.corrcoef(Y[test_mask], pred_y)[0, 1]
        pcc_scores.append(pcc)
        iter_results.append({
            'iteration': i + 1,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'n_test_cells': len(test_cells),
            'pcc': float(pcc),
        })
        print(f'  Iteration {i+1:2d}/{args.zs_iters}  '
              f'train={train_mask.sum():5d}  test={test_mask.sum():5d}  '
              f'PCC: {pcc:.4f}  (held out {len(test_cells)} cell lines, drug seen)')

    pcc_arr = np.array(pcc_scores)
    print(f'\nZero-shot gene-only (cell line) PCC over {len(pcc_arr)} iterations:')
    print(f'  Mean:   {pcc_arr.mean():.4f}')
    print(f'  Std:    {pcc_arr.std():.4f}')
    print(f'  Min:    {pcc_arr.min():.4f}')
    print(f'  Max:    {pcc_arr.max():.4f}')

    results = {
        'test_frac': args.test_frac,
        'pcc_scores': pcc_arr,
        'iterations': iter_results,
    }

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved zero-shot gene-only results to {result_out}')


def zero_shot_xgboost_gene_only_evaluation(args):
    print(f'Loading preprocessed data from {args.processed_data_path}...')
    with open(args.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    delta_expr = data['delta_expr']
    viability = data['viability']
    durations = data['durations']
    cell_ids = np.array(data['cell_ids'])
    drug_names = np.array(data['drug_names'])

    print(f'  Total samples:       {len(viability)}')
    print(f'  Unique drugs:        {len(set(drug_names))}')
    print(f'  Unique cell lines:   {len(set(cell_ids))}')
    print(f'  Timepoints present:  {sorted(set(durations))}')
    print(f'  Split by:            {args.split_by}')
    print(f'  Test fraction:       {args.test_frac}')
    print(f'  Iterations:          {args.zs_iters}')

    duration_col = durations.reshape(-1, 1).astype(np.float32)
    X = np.concatenate([delta_expr, duration_col], axis=1).astype(np.float32)
    Y = viability.astype(np.float32)
    print(f'\nFeature matrix X shape: {X.shape}  (delta_expr + duration)')

    unique_cell_lines = np.array(sorted(set(cell_ids)))

    np.random.seed(args.seed)
    pcc_scores = []
    iter_results = []

    for i in range(args.zs_iters):
        n_test_cells = max(1, int(len(unique_cell_lines) * args.test_frac))
        test_cells = set(np.random.choice(unique_cell_lines, n_test_cells, replace=False))
        train_cells = set(unique_cell_lines) - test_cells

        train_mask = np.array([c in train_cells for c in cell_ids])
        train_drugs = set(drug_names[train_mask])
        test_mask = np.array([c in test_cells and d in train_drugs
                               for c, d in zip(cell_ids, drug_names)])

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(f'  Iteration {i+1:2d}/{args.zs_iters}  skipped (empty split)')
            continue

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            tree_method='hist',
            device=args.device,
            random_state=args.seed + i,
            verbosity=0,
        )
        model.fit(X[train_mask], Y[train_mask])
        pred_y = model.predict(X[test_mask])

        pcc = np.corrcoef(Y[test_mask], pred_y)[0, 1]
        pcc_scores.append(pcc)
        iter_results.append({
            'iteration': i + 1,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
            'n_test_cells': len(test_cells),
            'pcc': float(pcc),
        })
        print(f'  Iteration {i+1:2d}/{args.zs_iters}  '
              f'train={train_mask.sum():5d}  test={test_mask.sum():5d}  '
              f'PCC: {pcc:.4f}  (held out {len(test_cells)} cell lines, drug seen)')

    pcc_arr = np.array(pcc_scores)
    print(f'\nZero-shot XGBoost gene-only (cell line) PCC over {len(pcc_arr)} iterations:')
    print(f'  Mean:   {pcc_arr.mean():.4f}')
    print(f'  Std:    {pcc_arr.std():.4f}')
    print(f'  Min:    {pcc_arr.min():.4f}')
    print(f'  Max:    {pcc_arr.max():.4f}')

    results = {
        'test_frac': args.test_frac,
        'pcc_scores': pcc_arr,
        'iterations': iter_results,
    }

    result_out = Path(args.result_out)
    result_out.parent.mkdir(parents=True, exist_ok=True)
    with open(result_out, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved zero-shot XGBoost gene-only results to {result_out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Ridge regression cell viability model on CTRPv2 + LINCS data'
    )
    parser.add_argument('--processed_data_path', type=str, required=True, default='/athena/angsd/scratch/ssl4003/sequential_drug_combination/pairwise/data/cell_viability_data/processed_ctrpv2_lincs_delta_expr.pkl',
                        help='Path to .pkl file from preprocess_ctrpv2.py')
    parser.add_argument('--model_out', type=str, required=True,
                        help='Output path for trained model bundle (.pkl)')
    parser.add_argument('--result_out', type=str, required=True,
                        help='Output path for PCC results (.pkl)')
    parser.add_argument('--model_type', type=str, default='ridge',
                        choices=['ridge', 'ridge_gene_only', 'xgboost', 'xgboost_gene_only',
                                 'gridsearch_xgboost', 'gridsearch_xgboost_gene_only',
                                 'zero_shot', 'zero_shot_xgboost',
                                 'zero_shot_gene_only', 'zero_shot_xgboost_gene_only'],
                        help='Model type to train (default: ridge)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge regularization strength (default: 1.0).')
    parser.add_argument('--n_estimators', type=int, default=500,
                        help='XGBoost: number of trees (default: 500)')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='XGBoost: max tree depth (default: 6)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='XGBoost: learning rate (default: 0.05)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='XGBoost: row subsampling ratio (default: 0.8)')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='XGBoost: column subsampling ratio per tree (default: 0.8)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='XGBoost device: cpu or cuda (default: cpu)')
    parser.add_argument('--gs_folds', type=int, default=5,
                        help='GridSearch: number of CV folds (default: 5)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='GridSearch: parallel jobs, -1 = all cores (default: -1)')
    parser.add_argument('--split_by', type=str, default='drug', choices=['drug', 'cell_line', 'both'],
                        help='Zero-shot: entity type to hold out (default: drug)')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Zero-shot: fraction of cell lines to hold out as test (default: 0.2)')
    parser.add_argument('--zs_iters', type=int, default=20,
                        help='Zero-shot: number of random hold-out iterations (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    if args.model_type == 'ridge_gene_only':
        generate_gene_only_model(args)
    elif args.model_type == 'xgboost_gene_only':
        generate_xgboost_gene_only_model(args)
    elif args.model_type == 'xgboost':
        generate_xgboost_model(args)
    elif args.model_type == 'gridsearch_xgboost':
        gridsearch_xgboost_model(args)
    elif args.model_type == 'gridsearch_xgboost_gene_only':
        gridsearch_xgboost_gene_only_model(args)
    elif args.model_type == 'zero_shot':
        zero_shot_evaluation(args)
    elif args.model_type == 'zero_shot_xgboost':
        zero_shot_xgboost_evaluation(args)
    elif args.model_type == 'zero_shot_gene_only':
        zero_shot_gene_only_evaluation(args)
    elif args.model_type == 'zero_shot_xgboost_gene_only':
        zero_shot_xgboost_gene_only_evaluation(args)
    else:
        generate_cell_viability_model(args)
