import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from ifood.model.train import LGBMTrainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


class LGBMHyperparameterOptimizer:
    def __init__(self, trainer: LGBMTrainer, n_trials: int = 50):
        self.trainer = trainer
        self.n_trials = n_trials
        self.study = None

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        X_train, _, y_train, _ = self.trainer.prepare_data()
        kf = KFold(n_splits=3, shuffle=True, random_state=self.trainer.random_state)
        rmses = []
        r2s = []
        spearmans = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = lgb.LGBMRegressor(
                **params, random_state=self.trainer.random_state, verbose=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - preds) ** 2))
            r2 = r2_score(y_val, preds)
            spearman_corr, _ = spearmanr(y_val, preds)
            rmses.append(rmse)
            r2s.append(r2)
            spearmans.append(spearman_corr)
        mean_rmse = np.mean(rmses)
        trial.set_user_attr("r2", np.mean(r2s))
        trial.set_user_attr("spearman_correlation", np.mean(spearmans))
        return mean_rmse

    def optimize(self) -> optuna.Study:
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.trainer.random_state),
        )
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.study
