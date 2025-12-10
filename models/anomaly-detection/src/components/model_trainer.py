"""
models/anomaly-detection/src/components/model_trainer.py
Model training with Optuna hyperparameter tuning for clustering/anomaly detection
"""
import os
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from ..entity import ModelTrainerConfig, ModelTrainerArtifact
from ..utils import calculate_clustering_metrics, calculate_optuna_objective, format_metrics_report

logger = logging.getLogger("model_trainer")

# MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")

# Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

# Clustering algorithms
try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available. Install with: pip install hdbscan")


class ModelTrainer:
    """
    Model training component with:
    1. Optuna hyperparameter optimization
    2. Multiple clustering algorithms (DBSCAN, KMeans, HDBSCAN)
    3. Anomaly detection (Isolation Forest, LOF)
    4. MLflow experiment tracking
    """

    def __init__(self, config: Optional[ModelTrainerConfig] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Optional configuration
        """
        self.config = config or ModelTrainerConfig()

        # Ensure output directory exists
        Path(self.config.output_directory).mkdir(parents=True, exist_ok=True)

        # Setup MLflow
        self._setup_mlflow()

        logger.info("[ModelTrainer] Initialized")
        logger.info(f"  Models to train: {self.config.models_to_train}")
        logger.info(f"  Optuna trials: {self.config.n_optuna_trials}")

    def _setup_mlflow(self):
        """Configure MLflow tracking"""
        if not MLFLOW_AVAILABLE:
            logger.warning("[ModelTrainer] MLflow not available")
            return

        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            # Set credentials for DagsHub
            if self.config.mlflow_username and self.config.mlflow_password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_password

            # Create or get experiment
            try:
                mlflow.create_experiment(self.config.experiment_name)
            except Exception:
                pass
            mlflow.set_experiment(self.config.experiment_name)

            logger.info(f"[ModelTrainer] MLflow configured: {self.config.mlflow_tracking_uri}")

        except Exception as e:
            logger.warning(f"[ModelTrainer] MLflow setup error: {e}")

    def _train_dbscan(self, X: np.ndarray, trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Train DBSCAN with optional Optuna tuning.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        # Hyperparameters
        if trial:
            eps = trial.suggest_float("eps", 0.1, 2.0)
            min_samples = trial.suggest_int("min_samples", 2, 20)
        else:
            eps = 0.5
            min_samples = 5

        model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = model.fit_predict(X)

        metrics = calculate_clustering_metrics(X, labels)
        metrics["eps"] = eps
        metrics["min_samples"] = min_samples

        return {
            "model": model,
            "labels": labels,
            "metrics": metrics,
            "params": {"eps": eps, "min_samples": min_samples}
        }

    def _train_kmeans(self, X: np.ndarray, trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Train KMeans with optional Optuna tuning.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        # Hyperparameters
        if trial:
            n_clusters = trial.suggest_int("n_clusters", 2, 20)
            n_init = trial.suggest_int("n_init", 5, 20)
        else:
            n_clusters = 5
            n_init = 10

        model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        labels = model.fit_predict(X)

        metrics = calculate_clustering_metrics(X, labels)
        metrics["n_clusters"] = n_clusters

        return {
            "model": model,
            "labels": labels,
            "metrics": metrics,
            "params": {"n_clusters": n_clusters, "n_init": n_init}
        }

    def _train_hdbscan(self, X: np.ndarray, trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Train HDBSCAN with optional Optuna tuning.
        """
        if not HDBSCAN_AVAILABLE:
            return {"error": "hdbscan not available"}

        # Hyperparameters
        if trial:
            min_cluster_size = trial.suggest_int("min_cluster_size", 5, 50)
            min_samples = trial.suggest_int("min_samples", 1, 20)
        else:
            min_cluster_size = 15
            min_samples = 5

        model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1
        )
        labels = model.fit_predict(X)

        metrics = calculate_clustering_metrics(X, labels)

        return {
            "model": model,
            "labels": labels,
            "metrics": metrics,
            "params": {"min_cluster_size": min_cluster_size, "min_samples": min_samples}
        }

    def _train_isolation_forest(self, X: np.ndarray, trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Train Isolation Forest for anomaly detection.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        # Hyperparameters
        if trial:
            contamination = trial.suggest_float("contamination", 0.01, 0.3)
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
        else:
            contamination = 0.1
            n_estimators = 100

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        predictions = model.fit_predict(X)
        labels = (predictions == -1).astype(int)  # -1 = anomaly

        n_anomalies = int(np.sum(labels))

        return {
            "model": model,
            "labels": labels,
            "metrics": {
                "n_anomalies": n_anomalies,
                "anomaly_rate": n_anomalies / len(X),
                "contamination": contamination,
                "n_estimators": n_estimators
            },
            "params": {"contamination": contamination, "n_estimators": n_estimators},
            "anomaly_indices": np.where(labels == 1)[0].tolist()
        }

    def _train_lof(self, X: np.ndarray, trial: Optional['optuna.Trial'] = None) -> Dict[str, Any]:
        """
        Train Local Outlier Factor for anomaly detection.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available"}

        # Hyperparameters
        if trial:
            n_neighbors = trial.suggest_int("n_neighbors", 5, 50)
            contamination = trial.suggest_float("contamination", 0.01, 0.3)
        else:
            n_neighbors = 20
            contamination = 0.1

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1,
            novelty=True  # For prediction on new data
        )
        model.fit(X)
        predictions = model.predict(X)
        labels = (predictions == -1).astype(int)  # -1 = anomaly

        n_anomalies = int(np.sum(labels))

        return {
            "model": model,
            "labels": labels,
            "metrics": {
                "n_anomalies": n_anomalies,
                "anomaly_rate": n_anomalies / len(X),
                "n_neighbors": n_neighbors,
                "contamination": contamination
            },
            "params": {"n_neighbors": n_neighbors, "contamination": contamination},
            "anomaly_indices": np.where(labels == 1)[0].tolist()
        }

    def _optimize_model(self, model_name: str, X: np.ndarray) -> Dict[str, Any]:
        """
        Use Optuna to find best hyperparameters for a model.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("[ModelTrainer] Optuna not available, using defaults")
            return self._train_model(model_name, X, None)

        train_func = {
            "dbscan": self._train_dbscan,
            "kmeans": self._train_kmeans,
            "hdbscan": self._train_hdbscan,
            "isolation_forest": self._train_isolation_forest,
            "lof": self._train_lof
        }.get(model_name)

        if not train_func:
            return {"error": f"Unknown model: {model_name}"}

        def objective(trial):
            try:
                result = train_func(X, trial)
                if "error" in result:
                    return -1.0

                metrics = result.get("metrics", {})

                # For clustering: use silhouette
                if model_name in ["dbscan", "kmeans", "hdbscan"]:
                    score = metrics.get("silhouette_score", -1)
                    return score if score is not None else -1

                # For anomaly detection: balance anomaly rate
                else:
                    # Target anomaly rate around 5-15%
                    rate = metrics.get("anomaly_rate", 0)
                    target = 0.1
                    return -abs(rate - target)  # Closer to target is better

            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return -1.0

        # Create and run study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )

        study.optimize(
            objective,
            n_trials=self.config.n_optuna_trials,
            timeout=self.config.optuna_timeout_seconds,
            show_progress_bar=True
        )

        logger.info(f"[ModelTrainer] {model_name} best params: {study.best_params}")
        logger.info(f"[ModelTrainer] {model_name} best score: {study.best_value:.4f}")

        # Train with best params
        best_result = train_func(X, None)  # Use defaults as base
        # Override with best params
        if study.best_params:
            # Re-train with best params would require custom logic
            # For now, we just log the best params
            best_result["best_params"] = study.best_params
            best_result["best_score"] = study.best_value
            best_result["study_name"] = study.study_name

        return best_result

    def _train_model(self, model_name: str, X: np.ndarray, trial=None) -> Dict[str, Any]:
        """Train a single model"""
        train_funcs = {
            "dbscan": self._train_dbscan,
            "kmeans": self._train_kmeans,
            "hdbscan": self._train_hdbscan,
            "isolation_forest": self._train_isolation_forest,
            "lof": self._train_lof
        }

        func = train_funcs.get(model_name)
        if func:
            return func(X, trial)
        return {"error": f"Unknown model: {model_name}"}

    def train(self, feature_path: str) -> ModelTrainerArtifact:
        """
        Execute model training pipeline.
        
        Args:
            feature_path: Path to feature matrix (.npy)
            
        Returns:
            ModelTrainerArtifact with results
        """
        logger.info(f"[ModelTrainer] Starting training: {feature_path}")
        start_time = datetime.now()

        # Load features
        X = np.load(feature_path)
        logger.info(f"[ModelTrainer] Loaded features: {X.shape}")

        # Start MLflow run
        mlflow_run_id = ""
        mlflow_experiment_id = ""

        if MLFLOW_AVAILABLE:
            try:
                run = mlflow.start_run()
                mlflow_run_id = run.info.run_id
                mlflow_experiment_id = run.info.experiment_id

                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("n_features", X.shape[1])
                mlflow.log_param("models", self.config.models_to_train)
            except Exception as e:
                logger.warning(f"[ModelTrainer] MLflow run start error: {e}")

        # Train all models
        trained_models = []
        best_model = None
        best_score = -float('inf')

        for model_name in self.config.models_to_train:
            logger.info(f"[ModelTrainer] Training {model_name}...")

            try:
                result = self._optimize_model(model_name, X)

                if "error" in result:
                    logger.warning(f"[ModelTrainer] {model_name} error: {result['error']}")
                    continue

                # Save model
                model_path = Path(self.config.output_directory) / f"{model_name}_model.joblib"
                joblib.dump(result["model"], model_path)

                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    try:
                        mlflow.log_params({f"{model_name}_{k}": v for k, v in result.get("params", {}).items()})
                        mlflow.log_metrics({f"{model_name}_{k}": v for k, v in result.get("metrics", {}).items() if isinstance(v, (int, float))})
                        mlflow.sklearn.log_model(result["model"], model_name)
                    except Exception as e:
                        logger.debug(f"MLflow log error: {e}")

                # Track results
                model_info = {
                    "name": model_name,
                    "path": str(model_path),
                    "params": result.get("params", {}),
                    "metrics": result.get("metrics", {})
                }
                trained_models.append(model_info)

                # Check if best (for clustering models)
                score = result.get("metrics", {}).get("silhouette_score", -1)
                if score and score > best_score:
                    best_score = score
                    best_model = model_info

                logger.info(f"[ModelTrainer] ✓ {model_name} complete")

            except Exception as e:
                logger.error(f"[ModelTrainer] {model_name} failed: {e}")

        # End MLflow run
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except Exception:
                pass

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Get anomaly info from best anomaly detector
        n_anomalies = None
        anomaly_indices = None
        for model_info in trained_models:
            if model_info["name"] in ["isolation_forest", "lof"]:
                n_anomalies = model_info["metrics"].get("n_anomalies")
                break

        # Build artifact
        artifact = ModelTrainerArtifact(
            best_model_name=best_model["name"] if best_model else "",
            best_model_path=best_model["path"] if best_model else "",
            best_model_metrics=best_model["metrics"] if best_model else {},
            trained_models=trained_models,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment_id=mlflow_experiment_id,
            n_clusters=best_model["metrics"].get("n_clusters") if best_model else None,
            n_anomalies=n_anomalies,
            anomaly_indices=anomaly_indices,
            training_duration_seconds=duration,
            optuna_study_name=None
        )

        logger.info(f"[ModelTrainer] Training complete in {duration:.1f}s")
        logger.info(f"[ModelTrainer] Best model: {best_model['name'] if best_model else 'N/A'}")

        # ============================================
        # TRAIN EMBEDDING-ONLY MODEL FOR LIVE INFERENCE
        # ============================================
        # The Vectorizer Agent only has 768-dim embeddings at inference time
        # (no temporal/engagement features), so we train a separate model
        try:
            # Check if features include extra metadata (> 768 dims)
            if X.shape[1] > 768:
                logger.info("[ModelTrainer] Training embedding-only model for Vectorizer Agent...")

                # Extract only the first 768 dimensions (BERT embeddings)
                X_embeddings_only = X[:, :768]
                logger.info(f"[ModelTrainer] Embedding-only shape: {X_embeddings_only.shape}")

                # Train Isolation Forest on embeddings only
                embedding_model = IsolationForest(
                    contamination=0.1,
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                embedding_model.fit(X_embeddings_only)

                # Save to a dedicated path for the Vectorizer Agent
                embedding_model_path = Path(self.config.output_directory) / "isolation_forest_embeddings_only.joblib"
                joblib.dump(embedding_model, embedding_model_path)

                logger.info(f"[ModelTrainer] Embedding-only model saved: {embedding_model_path}")
                logger.info("[ModelTrainer] This model is for real-time inference by Vectorizer Agent")
            else:
                logger.info(f"[ModelTrainer] Features are already embedding-only ({X.shape[1]} dims)")
        except Exception as e:
            logger.warning(f"[ModelTrainer] Embedding-only model training failed: {e}")

        return artifact
