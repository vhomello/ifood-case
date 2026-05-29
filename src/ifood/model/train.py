import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from typing import List


class LGBMTrainer:
    def __init__(
        self,
        data_path: str,
        random_state: int,
        target: str,
        features: List[str],
        categorical_features: List[str],
        lgbm_params: dict = None,
    ):
        self.data_path = data_path
        self.random_state = random_state
        self.target = target
        self.features = features
        self.categorical_features = categorical_features
        self.lgbm_params = lgbm_params
        self.model = None

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Loads the parquet dataset and filters it to include only split == 'train'.
        Applies necessary preprocessing to feature columns.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        df = pd.read_parquet(self.data_path)

        for col in self.categorical_features:
            df[col] = df[col].fillna("None").astype("category")

        return df

    def prepare_data(self):
        """
        Loads preprocessed training data and splits it into train/validation sets.

        Returns:
            X_train, X_val, y_train, y_val
        """
        df = self.load_and_preprocess()
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]

        X_train = df_train[self.features]
        X_test = df_test[self.features]
        y_train = df_train[self.target]
        y_test = df_test[self.target]

        return X_train, X_test, y_train, y_test

    def train(self, params: dict = None) -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
        """
        Trains the LGBMRegressor on the training split and evaluates it on train and test splits.

        Args:
            params: Dictionary of hyperparameters for LGBMRegressor.

        Returns:
            pd.DataFrame: Training and testing metrics (RMSE, R2, Spearman Correlation)
        """
        X_train, X_test, y_train, y_test = self.prepare_data()

        default_params = {
            "random_state": self.random_state,
            "n_estimators": 100,
            "learning_rate": 0.05,
            "verbose": -1,
        }

        if params is not None:
            default_params.update(params)

        self.model = lgb.LGBMRegressor(**default_params)

        self.model.fit(
            X_train,
            y_train,
        )

        train_metrics = self.evaluate(X_train, y_train)
        test_metrics = self.evaluate(X_test, y_test)

        return self.model, pd.DataFrame({"train": train_metrics, "test": test_metrics})

    def evaluate(self, X_eval: pd.DataFrame, y_eval: pd.Series) -> dict:
        """
        Computes validation metrics for the trained model.

        Args:
            X_eval: Evaluation feature set.
            y_eval: Evaluation target set.

        Returns:
            dict: Evaluation metrics (RMSE, R2, Spearman Correlation)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        preds = self.model.predict(X_eval)

        mse = np.mean((y_eval - preds) ** 2)
        rmse = np.sqrt(mse)

        r2 = r2_score(y_eval, preds)

        spearman_corr, _ = spearmanr(y_eval, preds)

        return {"rmse": rmse, "r2": r2, "spearman_correlation": spearman_corr}
