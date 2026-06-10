import os
import pickle

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 20968061

# cargar datos
df = pd.read_csv("water_potability.csv").dropna()
X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model


def optimize_model():
    # crear experimento con nombre interpretable
    experiment_name = "XGBoost-Water-Potability"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        run_name = f"XGBoost lr={params['learning_rate']:.4f} depth={params['max_depth']}"

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name):
            model = XGBClassifier(
                **params,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            )
            model.fit(X_train, y_train)

            # registrar hiperparámetros y métrica
            mlflow.log_params(params)
            f1 = f1_score(y_test, model.predict(X_test))
            mlflow.log_metric("valid_f1", f1)

            # registrar modelo
            mlflow.sklearn.log_model(model, "model")

        return f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # obtener y guardar el mejor modelo
    best_model = get_best_model(experiment.experiment_id)
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Mejor F1: {study.best_value:.4f}")
    print("Modelo guardado en models/best_model.pkl")


if __name__ == "__main__":
    optimize_model()
