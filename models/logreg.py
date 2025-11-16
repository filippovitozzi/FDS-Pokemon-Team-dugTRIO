import pandas as pd
from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from models.metrics import test_model


class LogReg:

    estimator = None


    def __init__(self, random_state = 42, pipe=None, param_grid_list=None):
        self.random_state = random_state
        if pipe is None:
            self.pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=5000, random_state=self.random_state))
            ])
        else:
            self.pipe = pipe
        
        if param_grid_list is None:
            param_grid_list = []
            penalties = ["l1", "l2"]
            solvers = ["liblinear", "saga", "lbfgs"]
            for penalty, solver in product(penalties, solvers):
                if penalty == "l1" and solver not in ["liblinear", "saga"]:
                    continue
                if penalty == "l2" and solver not in ["liblinear", "lbfgs", "saga"]:
                    continue
                param_grid_list.append({
                    "logreg__penalty": [penalty],
                    "logreg__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50],
                    "logreg__solver": [solver],
                    "logreg__class_weight": [None, "balanced"]
                })
            self.param_grid = param_grid_list
        else:
            self.param_grid = param_grid_list
        
    def fit(self,X,y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(self.pipe, self.param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
        grid.fit(X, y)
        self.estimator = grid.best_estimator_

    def predict(self,X):
        if self.estimator is None:
            print("Cannot predict, call fit() first!")
            return None
        y_pred = self.estimator.predict(X)
        return y_pred

    def generate_submission(self, X_test, X_full, path="logreg_submission.csv"):
        test_pred_proba = self.estimator.predict_proba(X_test)[:, 1]
        test_pred_binary = (test_pred_proba >= 0.5).astype(int)
        submission_df = pd.DataFrame({
            "battle_id": X_full["battle_id"],
            "player_won": test_pred_binary
        })
        submission_df.to_csv(path, index=False)
        print(f"LogReg submission stored in: {path}")

    def get_performance_metrics(self,X,y_true):
        if not self.estimator:
            print("Cannot get_performance_metrics(), run fit() first!")
        return test_model(self.estimator,X,y_true)