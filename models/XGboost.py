import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from models.metrics import test_model

class XGBoost:

    def __init__(self, random_state = 42):
        self.xgb_base = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state
        )
        self.random_state = random_state
    
    def search_best_params(self, X, y, param_dist = None):
        if param_dist is None:
            param_dist = {
                "n_estimators": [600, 800, 1000],
                "learning_rate": [0.01, 0.015, 0.02],
                "max_depth": [3, 4, 5],
                "min_child_weight": [3, 5, 7],
                "subsample": [0.75, 0.8, 0.85],
                "colsample_bytree": [0.6, 0.7, 0.8],
                "gamma": [0, 0.5, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1.0, 1.5, 2.0],
            }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rand = RandomizedSearchCV(
            estimator=self.xgb_base,
            param_distributions=param_dist,
            n_iter=40,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        rand.fit(X, y)
        self.best_params = rand.best_params_
        self.best_score = rand.best_score_
        return self.best_params
    
    def refit(self,X_train,y_train,X_valid, y_valid):
        if not self.best_params:
            print("Best params not found, run search_best_params() first!")
            return
        self.best_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
            early_stopping_rounds=80,
            **self.best_params
        )

        self.best_xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        return

    def final_fit(self,X,y):
        if not self.best_params:
            print("Best params not found, run search_best_params() first!")
            return
        self.final_xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state,
            **self.best_params
        )
        self.final_xgb.fit(X, y)
        return
    
    def generate_submission(self, X_test, X_full, path="XGBoost_submission.csv"):
        if not self.final_xgb:
            print("Run final_fit() before generate_submission()!")
            return
        test_pred_proba = self.final_xgb.predict_proba(X_test)[:, 1]
        test_pred_binary = (test_pred_proba >= 0.5).astype(int)
        submission_df = pd.DataFrame({
            "battle_id": X_full["battle_id"],
            "player_won": test_pred_binary
        })
        submission_df.to_csv(path, index=False)
        print(f"XGBoost submission stored in: {path}")
        print(f"XGB submission (first 5 rows):\n{submission_df.head()}")

        
    def best_xgb_report(self, X_train, y_train, X_valid, y_valid):
        if not self.best_xgb:
            print("Run refit() before best_xgb_report()!")

        train_metrics = test_model(self.best_xgb, X_train, y_train)
        vaild_metrics = test_model(self.best_xgb, X_valid, y_valid)

        print("---------------------------TRAINING PERFORMANCE----------------------------")
        print(train_metrics)
        print("---------------------------VALIDATION PERFORMANCE----------------------------")
        print(vaild_metrics)
        # train_proba = self.best_xgb.predict_proba(X_train)[:, 1]
        # valid_proba = self.best_xgb.predict_proba(X_valid)[:, 1]

        # train_pred = (train_proba >= 0.5).astype(int)
        # valid_pred = (valid_proba >= 0.5).astype(int)

        # print("\n PERFORMANCE REPORT")
        # print(f"Training AUC: {roc_auc_score(y_train, train_proba):.4f} | ACC: {accuracy_score(y_train, train_pred):.4f}")
        # print(f"Validation AUC: {roc_auc_score(y_valid, valid_proba):.4f} | ACC: {accuracy_score(y_valid, valid_pred):.4f}")

    def get_performance_metrics(self,X,y_true):
        if not self.final_xgb:
            print("Cannot get_performance_metrics(), run final_fit() first!")
        return test_model(self.final_xgb,X,y_true)