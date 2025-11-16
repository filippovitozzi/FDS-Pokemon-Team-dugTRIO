
from sklearn.ensemble import VotingClassifier
import pandas as pd
from models.metrics import test_model

class Ensemble:

    def __init__(self, estimators=[], voting="soft"):
        self.estimator = VotingClassifier(estimators=estimators, voting=voting)


    def generate_submission(self, X_test, X_full, path="ensemble_submission.csv"):
        test_pred_proba = self.estimator.predict_proba(X_test)[:, 1]
        test_pred_binary = (test_pred_proba >= 0.5).astype(int)
        submission_df = pd.DataFrame({
            "battle_id": X_full["battle_id"],
            "player_won": test_pred_binary
        })
        submission_df.to_csv(path, index=False)
        print(f"Ensemble submission stored in: {path}")


    def fit(self, X, y):
        self.estimator.fit(X,y)

    def get_performance_metrics(self,X,y_true):
        if not self.estimator:
            print("Cannot get_performance_metrics(), run fit() first!")
        return test_model(self.estimator,X,y_true)
