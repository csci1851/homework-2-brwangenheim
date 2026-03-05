"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,   
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
        ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """

        # TODO: Implement train/test split and track feature names

        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if self.task == "classification" else None
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # TODO: Create classifier/regressor based on task and fit it
        if self.task == "classification":
            self.model = GradientBoostingClassifier(**self.params)
        else: # already guaranteed to be regression due to check in init
            self.model = GradientBoostingRegressor(**self.params)

        if self.use_scaler:
            self.model = Pipeline([
                ("scaler", self.scaler),
                ("model", self.model)
            ]) 
        self.model.fit(X_train, y_train)

        if verbose:
            print("Completed training gradient boosting")

        return self

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # TODO: Apply scaler when enabled, then predict
        if self.task == "classification" and return_proba:
            return self.model.predict_proba(X)

        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)

        if self.task == "classification":
            y_proba = self.predict(X_test, return_proba=True)

            # ensure y_test is 1D for roc_auc_score
            if isinstance(y_test, pd.DataFrame) or (len(np.shape(y_test)) > 1 and np.shape(y_test)[1] > 1):
                y_test = np.argmax(np.array(y_test), axis=1)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="macro"),
                "recall": recall_score(y_test, y_pred, average="macro"),
                "f1": f1_score(y_test, y_pred, average="macro"),
            }

            # ROC-AUC
            if y_proba.shape[1] == 2:  # binary
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:  # multi-class
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        else:  # regression
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), 
                "mae": mean_absolute_error(y_test, y_pred), 
                "r2": r2_score(y_test, y_pred)
            }

        return metrics
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        # TODO: Choose scoring metrics based on classification vs regression
        if self.task == "classification":
            model = GradientBoostingClassifier(**self.params)
            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        else:
            model = GradientBoostingRegressor(**self.params)
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

        if self.use_scaler:
            model = Pipeline([
                ("scaler", self.scaler),
                ("model", model)
            ])

        results = {}

        # TODO: Get mean, stdev of cross_val_score for each metric

        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            results[metric] = {
                "mean": scores.mean(),
                "std": scores.std(),
            }

        # absolute values for regression metrics
        if self.task == "regression":
            results["rmse"] = {
                "mean": np.sqrt(-results["neg_mean_squared_error"]["mean"]),
                "std": results["neg_mean_squared_error"]["std"],
            }
            results["mae"] = {
                "mean": -results["neg_mean_absolute_error"]["mean"],
                "std": results["neg_mean_absolute_error"]["std"],
            }

        return results


    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
        ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """

        # TODO: Optionally plot a bar chart of top_n feature importances
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if isinstance(self.model, Pipeline):
            model = self.model.named_steps["model"]
        else:
            model = self.model

        importances = model.feature_importances_

        df_importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        ).sort_values(by="importance", ascending=False)

        if plot:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df_importance.head(top_n),
                x="importance",
                y="feature",
            )
            plt.title("Top Feature Importances")
            plt.tight_layout()
            plt.show()

        return df_importance

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc", # changing this parameter manually
        ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task
        if self.task == "classification":
            model = GradientBoostingClassifier(random_state=self.params["random_state"])
        else:
            model = GradientBoostingRegressor(random_state=self.params["random_state"])

        if self.use_scaler:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", model),
                ]
            )
            param_grid = {f"model__{k}": v for k, v in param_grid.items()}

        # TODO: Initialize GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )

        # TODO: Perform grid search for hyperparameter tuning
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_ # for plotting boxes
        }

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
        ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """
        if isinstance(self.model, Pipeline):
            model = self.model.named_steps["model"]
        else:
            model = self.model

        if tree_index >= len(model.estimators_):
            raise IndexError("Tree index out of range.")

        tree = model.estimators_[tree_index, 0]

        plt.figure(figsize=figsize)
        plot_tree(tree, feature_names=self.feature_names, filled=True)
        plt.show()
