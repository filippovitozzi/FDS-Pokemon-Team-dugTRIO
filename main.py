import numpy as np
import os
import json
import pandas as pd
from helpers import get_pokemon_stats, get_all_statuses
from features import build_features
from models.ensemble import Ensemble

RANDOM_SEED = 66

for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data_directory="kaggle/input/fds-pokemon-battles-prediction-2025"
train_path=os.path.join(data_directory,"train.jsonl")
test_path=os.path.join(data_directory,"test.jsonl")

def load_jsonl(path,n_rows=None):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n_rows and i >= n_rows:
                break
            data.append(json.loads(line))
    return data
data_train=load_jsonl(train_path)
data_test=load_jsonl(test_path)

train_df=pd.DataFrame(data_train)
test_df=pd.DataFrame(data_test)

types = {
    "bulbasaur": ["grass", "poison"], "ivysaur": ["grass", "poison"], "venusaur": ["grass", "poison"],
    "charmander": "fire", "charmeleon": "fire", "charizard": ["fire", "flying"],
    "squirtle": "water", "wartortle": "water", "blastoise": "water",
    "caterpie": "bug", "metapod": "bug", "butterfree": ["bug", "flying"],
    "weedle": ["poison", "bug"], "kakuna": ["poison", "bug"], "beedrill": ["poison", "bug"],
    "pidgey": ["normal", "flying"], "pidgeotto": ["normal", "flying"], "pidgeot": ["normal", "flying"],
    "rattata": "normal", "raticate": "normal", "spearow": ["normal", "flying"],
    "fearow": ["normal", "flying"], "ekans": "poison", "arbok": "poison",
    "pikachu": "electric", "raichu": "electric", "sandshrew": "ground", "sandslash": "ground",
    "nidoran♀": "poison", "nidorina": "poison", "nidoqueen": ["poison", "ground"],
    "nidoran♂": "poison", "nidorino": "poison", "nidoking": ["poison", "ground"],
    "clefairy": "normal", "clefable": "normal", "vulpix": "fire", "ninetales": "fire",
    "zubat": ["poison", "flying"], "golbat": ["poison", "flying"], "oddish": ["grass", "poison"],
    "gloom": ["grass", "poison"], "vileplume": ["grass", "poison"], "paras": ["bug", "grass"],
    "parasect": ["bug", "grass"], "venonat": ["bug", "poison"], "venomoth": ["bug", "poison"],
    "diglett": "ground", "dugtrio": "ground", "meowth": "normal", "persian": "normal",
    "psyduck": "water", "golduck": "water","mankey":"fighting","primeape":"fighting","growlithe":"fire",
    "arcanine":"fire","poliwag":"water","poliwhirl":"water","poliwrath":["water","fighting"],"abra":"psychic",
    "kadabra":"psychic","alakazam":"psychic","machop": "fighting", "machoke": "fighting",
    "machamp": "fighting", "bellsprout": ["grass", "poison"], "weepinbell": ["grass", "poison"],
    "victreebel": ["grass", "poison"], "tentacool": ["water", "poison"], "tentacruel": ["water", "poison"],
    "geodude": ["rock", "ground"], "graveler": ["rock", "ground"], "golem": ["rock", "ground"],
    "ponyta": "fire", "rapidash": "fire", "slowpoke": ["water", "psychic"], "slowbro": ["water", "psychic"],
    "magnemite": "electric", "magneton": "electric", "farfetch'd": ["normal", "flying"],
    "doduo": ["normal", "flying"], "dodrio": ["normal", "flying"], "seel": "water", "dewgong": ["water", "ice"],
    "grimer": "poison", "muk": "poison", "shellder": "water", "cloyster": ["water", "ice"],
    "gastly": ["ghost", "poison"], "haunter": ["ghost", "poison"], "gengar": ["ghost", "poison"],
    "onix": ["rock", "ground"], "drowzee": "psychic", "hypno": "psychic", "krabby": "water", "kingler": "water",
    "exeggcute": ["grass", "psychic"], "exeggutor": ["grass", "psychic"], "cubone": "ground", "marowak": "ground",
    "lickitung": "normal", "koffing": "poison", "weezing": "poison", "rhyhorn": ["rock", "ground"],
    "rhydon": ["rock", "ground"], "chansey": "normal", "tangela": "grass", "kangaskhan": "normal",
    "horsea": "water", "seadra": "water", "goldeen": "water", "seaking": "water",
    "staryu": "water", "starmie": ["water", "psychic"], "mr. mime": "psychic", "scyther": ["bug", "flying"],
    "jynx": ["ice", "psychic"], "electabuzz": "electric", "magmar": "fire", "pinsir": "bug",
    "tauros": "normal", "magikarp": "water", "gyarados": ["water", "flying"], "lapras": ["water", "ice"],
    "ditto": "normal", "eevee": "normal", "vaporeon": "water", "jolteon": "electric", "flareon": "fire",
    "porygon": "normal", "omanyte": ["rock", "water"], "omastar": ["rock", "water"], "kabuto": ["rock", "water"],
    "kabutops": ["rock", "water"], "aerodactyl": ["rock", "flying"], "mew": "psychic", "mewtwo": "psychic",
    "voltorb": "electric",
    "electrode": "electric",
    "hitmonlee": "fighting",
    "hitmonchan": "fighting",
    "articuno": ["ice", "flying"],
    "zapdos": ["electric", "flying"],
    "moltres": ["fire", "flying"],
    "dratini": "dragon",
    "dragonair": "dragon",
    "dragonite": ["dragon", "flying"],
    "snorlax": "normal"
}

attack_types = {
    "normal": {"rock": 0.5, "ghost": 0},
    "fire": {"grass": 2, "fire": 0.5, "water": 0.5, "bug": 2, "rock": 0.5, "ice": 2, "dragon": 0.5},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"water": 2, "fire": 0.5, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5},
    "ice": {"grass": 2, "ice": 0.5, "water": 0.5, "ground": 2, "flying": 2, "dragon": 2},
    "fighting": {"normal": 2, "rock": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "ghost": 0},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2},
    "flying": {"grass": 2, "electric": 0.5, "fighting": 2, "bug": 2, "rock": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5},
    "bug": {"grass": 2, "fire": 0.5, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2},
    "ghost": {"ghost": 2, "psychic": 0},
    "dragon": {"dragon": 2}
}

statuses = get_all_statuses(data_train)
pokemon_stats= get_pokemon_stats(data_train)

train_features = build_features(data_train,pokemon_stats,attack_types,types,is_train=True)
test_features= build_features(data_test,pokemon_stats,attack_types,types,is_train=False)
# print(train_features.head())
# print(train_features.shape)
# print(test_features.shape)

TARGET = "player_won"

FEATURE_COLS = [c for c in train_features.columns if c not in [TARGET, "battle_id"]]
X_train = train_features[FEATURE_COLS]
y_train = train_features[TARGET].astype(int)
X_test = test_features[FEATURE_COLS]


from models.logreg import LogReg

log_clf = LogReg()

log_clf.fit(X_train, y_train)
print("------------LOGISTIC REGRESSION BEST ESTIMATOR PERFORMANCE------------")
print(log_clf.get_performance_metrics(X_train, y_train))

log_clf.generate_submission(X_test, test_features)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


log_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
])

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True, random_state=RANDOM_SEED))
])

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
estimators=[
        ("lr", log_clf),
        ("svm", svm_clf),
        ("tree", tree_clf)]

voting_clf = Ensemble(estimators=estimators)

voting_clf.fit(X_train, y_train)
print("------------ENSEMBLE BEST ESTIMATOR PERFORMANCE------------")
print(voting_clf.get_performance_metrics(X_train, y_train))

voting_clf.generate_submission(X_test, test_features)



from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from models.XGboost import XGBoost


#Dataset Shuffle for XGBoost only
train_features = shuffle(train_features, random_state=RANDOM_SEED).reset_index(drop=True)

#Split training set into train and validation set
X_train_xgb, X_valid, y_train_xgb, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_SEED
)

xgboost_clf = XGBoost(RANDOM_SEED)
xgboost_clf.search_best_params(X_train_xgb, y_train_xgb)
xgboost_clf.refit(X_train_xgb, y_train_xgb, X_valid, y_valid)
xgboost_clf.best_xgb_report(X_train_xgb, y_train_xgb, X_valid, y_valid)
#Refit on the whole training set
xgboost_clf.final_fit(X_train,y_train)

print("------------XGBOOST BEST ESTIMATOR PERFORMANCE------------")
print(xgboost_clf.get_performance_metrics(X_train,y_train))

xgboost_clf.generate_submission(X_test, test_features)
