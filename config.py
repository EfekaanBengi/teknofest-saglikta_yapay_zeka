CHARGE_MAP = {
    'R': 1, 'K': 1,
    'D': -1, 'E': -1,
    'H': 0, 'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0,
    'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

HYDRO_MAP = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

HYPER_GRIDS = {
    'LogisticRegression': {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['saga'],
        'classifier__max_iter': [2000]
    },
    'RandomForest': {
        'classifier__max_depth': [3, 5, 8],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__n_estimators': [100, 200]
    },
    'LightGBM': {
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__reg_lambda': [1.0, 5.0, 10.0],
        'classifier__n_estimators': [100]
    },
    'XGBoost': {
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__reg_lambda': [1.0, 5.0, 10.0],
        'classifier__n_estimators': [100]
    },
    'LightGBM_CFTR': {
        'classifier__max_depth': [2, 3],
        'classifier__reg_lambda': [10.0, 50.0],
        'classifier__n_estimators': [50],
        'classifier__learning_rate': [0.05]
    }
}
