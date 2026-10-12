"""Entrena y evalúa un modelo parametrizado de calidad de vino tinto.

Migra el flujo del notebook baseline (carga, limpieza, pipeline, evaluación)
a un script ejecutable. Ridge extiende la regresión lineal del notebook
para poder comparar hiperparámetros de regularización sin editar el código.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "winequality-red.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TRACKING_DB = PROJECT_ROOT / "mlflow.db"
EXPERIMENT_NAME = "TC5061_Wine_Quality_Experimentacion"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Expone hiperparámetros y semilla como argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena Ridge sobre Wine Quality (vino tinto) y registra la "
            "corrida en MLflow Tracking."
        )
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fuerza de regularización L2 de Ridge. Valor por defecto: 1.0.",
    )
    parser.add_argument(
        "--fit-intercept",
        dest="fit_intercept",
        action="store_true",
        default=True,
        help="Incluye intercepto en el modelo. Activo por defecto.",
    )
    parser.add_argument(
        "--no-fit-intercept",
        dest="fit_intercept",
        action="store_false",
        help="Entrena Ridge sin intercepto.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        help="Algoritmo de solución de Ridge. Valor por defecto: auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla aleatoria para partición y solvers estocásticos. Por defecto: 42.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporción del conjunto de prueba. Valor por defecto: 0.20.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Ruta al CSV de vino tinto.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre descriptivo de la corrida en MLflow.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Fija semillas para que la misma configuración reproduzca el mismo resultado."""
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Carga el CSV original, separado por punto y coma."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró {data_path}. Coloca winequality-red.csv en data/."
        )
    return pd.read_csv(data_path, sep=";")


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la misma limpieza documentada en el notebook: quita duplicados exactos."""
    cleaned = raw_df.drop_duplicates().reset_index(drop=True)
    if cleaned.isna().sum().sum() != 0:
        raise ValueError("Tras la limpieza siguen existiendo valores faltantes.")
    return cleaned


def split_features(
    cleaned: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa predictores y objetivo con partición estratificada y semilla fija."""
    features = cleaned.drop(columns="quality")
    target = cleaned["quality"]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )


def build_pipeline(alpha: float, fit_intercept: bool, solver: str, seed: int) -> Pipeline:
    """Construye el pipeline: imputación, escalado y Ridge."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                Ridge(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    solver=solver,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE y R² sobre el conjunto de prueba."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_residual_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """Guarda la gráfica de residuos como artefacto visual de la corrida."""
    residuals = y_true.to_numpy() - y_pred
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_true, y_pred, alpha=0.6, color="#1f4e79")
    axes[0].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "--",
        color="black",
    )
    axes[0].set(
        xlabel="Calidad real",
        ylabel="Calidad estimada",
        title="Valores reales frente a estimados",
    )
    axes[1].scatter(y_pred, residuals, alpha=0.6, color="#8c1d40")
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set(
        xlabel="Calidad estimada",
        ylabel="Residuo",
        title="Residuos del modelo",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_metrics_file(metrics: dict[str, float], output_path: Path) -> Path:
    """Persiste las métricas en JSON para registrarlas como artefacto."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path


def log_run(
    args: argparse.Namespace,
    pipeline: Pipeline,
    metrics: dict[str, float],
    residual_path: Path,
    metrics_path: Path,
    n_rows: int,
    n_features: int,
    input_example: pd.DataFrame,
) -> str:
    """Registra parámetros, métricas, artefactos y modelo en MLflow."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(f"sqlite:///{DEFAULT_TRACKING_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = args.run_name or (
        f"ridge_alpha{args.alpha}_intercept{int(args.fit_intercept)}_{args.solver}_seed{args.seed}"
    )
    with mlflow.start_run(
        run_name=run_name,
        tags={"mlflow.user": "Carlos Ojeda"},
    ) as run:
        mlflow.log_params(
            {
                "alpha": args.alpha,
                "fit_intercept": args.fit_intercept,
                "solver": args.solver,
                "seed": args.seed,
                "test_size": args.test_size,
                "model": "Ridge",
                "imputer": "median",
                "scaler": "StandardScaler",
                "clean_rows": n_rows,
                "n_features": n_features,
            }
        )
        mlflow.log_metrics(
            {
                "test_mae": metrics["mae"],
                "test_rmse": metrics["rmse"],
                "test_r2": metrics["r2"],
            }
        )
        mlflow.set_tags(
            {
                "course": "TC5061",
                "student": "Carlos Ojeda",
                "problem": "regression",
                "dataset": "UCI Wine Quality red",
            }
        )
        mlflow.log_artifact(str(residual_path), artifact_path="plots")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.sklearn.log_model(
            pipeline,
            name="ridge_pipeline",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=input_example,
            pip_requirements=[
                "scikit-learn==1.9.1",
                "pandas==3.0.6",
                "numpy==2.5.3",
                "cloudpickle==3.1.2",
            ],
        )
        return run.info.run_id


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    raw_df = load_dataset(args.data_path)
    cleaned = clean_dataset(raw_df)
    x_train, x_test, y_train, y_test = split_features(
        cleaned, args.test_size, args.seed
    )
    pipeline = build_pipeline(args.alpha, args.fit_intercept, args.solver, args.seed)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = evaluate_regression(y_test, predictions)

    artifact_dir = DEFAULT_ARTIFACT_DIR / f"seed{args.seed}_alpha{args.alpha}"
    residual_path = save_residual_plot(
        y_test, predictions, artifact_dir / "residuals.png"
    )
    metrics_path = save_metrics_file(metrics, artifact_dir / "metrics.json")
    run_id = log_run(
        args=args,
        pipeline=pipeline,
        metrics=metrics,
        residual_path=residual_path,
        metrics_path=metrics_path,
        n_rows=len(cleaned),
        n_features=x_train.shape[1],
        input_example=x_train.head(3),
    )

    print(f"Run ID: {run_id}")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(
        f"MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
