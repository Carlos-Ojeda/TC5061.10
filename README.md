# TC5061 | Código de experimentación con MLflow

Estudiante: Carlos Ojeda  
Matrícula: A01741085  
Curso: TC5061 — MLOps

Este repositorio convierte el modelo baseline de la semana anterior (Wine Quality, vino tinto) en un script de entrenamiento parametrizado e instrumentado con MLflow Tracking.

El notebook se conserva solo para la exploración de datos. El entrenamiento, la evaluación y el registro de experimentos viven en `train.py`.

## Dataset

- Fuente: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Archivo local: `data/winequality-red.csv`
- Problema: regresión sobre `quality`
- Limpieza: se eliminan duplicados exactos (misma decisión documentada en el notebook)

## Requisitos

Python 3.11 o superior. En este trabajo se usó Python 3.13.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo ejecutar el script

El script corre con valores por defecto si no se pasa ningún argumento:

```bash
python train.py
```

Hiperparámetros expuestos:

| Argumento | Default | Descripción |
|---|---|---|
| `--alpha` | `1.0` | Regularización L2 de Ridge |
| `--fit-intercept` / `--no-fit-intercept` | intercepto activo | Incluye o no el intercepto |
| `--solver` | `auto` | Algoritmo de solución de Ridge |
| `--seed` | `42` | Semilla de la partición y del solver |
| `--test-size` | `0.20` | Proporción del conjunto de prueba |

## Corridas comparadas

Todas las corridas se agrupan en el experimento `TC5061_Wine_Quality_Experimentacion`.

```bash
python train.py --run-name ridge_default
python train.py --alpha 0.01 --run-name ridge_alpha_0.01
python train.py --alpha 10 --run-name ridge_alpha_10
python train.py --alpha 100 --run-name ridge_alpha_100
python train.py --no-fit-intercept --run-name ridge_sin_intercepto
python train.py --alpha 1 --solver saga --run-name ridge_solver_saga
python train.py --run-name ridge_default_repro
```

Mejor configuración observada: `--alpha 0.01` (MAE 0.5066, R² 0.3657). El hiperparámetro más influyente fue `fit_intercept`.

## Reproducibilidad

La semilla se fija en tres puntos:

1. `random.seed(seed)`
2. `numpy.random.seed(seed)`
3. `train_test_split(..., random_state=seed)` y `Ridge(..., random_state=seed)`

Dos ejecuciones con los mismos argumentos producen las mismas métricas. El tracking usa SQLite (`mlflow.db`) para evitar el FileStore bloqueado en MLflow 3.16.

## Interfaz de MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abre http://127.0.0.1:5000 y selecciona el experimento `TC5061_Wine_Quality_Experimentacion`. Cada corrida registra:

- parámetros (`alpha`, `fit_intercept`, `solver`, `seed`)
- métricas (`test_mae`, `test_rmse`, `test_r2`)
- artefactos (gráfica de residuos y `metrics.json`)
- el pipeline entrenado (`ridge_pipeline`)

## Estructura

```text
train.py
requirements.txt
README.md
data/winequality-red.csv
notebooks/TC5061_Notebook_Baseline_CarlosOjeda.ipynb
evidencias/
```
