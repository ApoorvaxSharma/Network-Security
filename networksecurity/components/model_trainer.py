import os
import sys
from urllib.parse import urlparse
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging 
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.main_utils.utils import evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.base import clone

import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric, best_model_name: str):
        if mlflow.active_run():
            mlflow.end_run()

        #mlflow.set_tracking_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run(run_name=f"Training_{best_model_name}"):
            # Log metrics
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall", classificationmetric.recall_score)

            # Log model
            #if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
            #else:
                #mlflow.sklearn.log_model(best_model, "model")

            # Optional: log preprocessing object
            mlflow.log_artifact(self.data_transformation_artifact.transformed_object_file_path)
    
            
        
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {'criterion': ['gini', 'entropy', 'log_loss']},
                "Random Forest": {'n_estimators': [8, 16, 32, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            model_report = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models, params=params
            )

            # Pick the best model based on accuracy
            best_model_name = max(model_report, key=lambda name: model_report[name]["accuracy"])
            best_model_score = model_report[best_model_name]["accuracy"]

            logging.info(f"Best model selected: {best_model_name} with accuracy {best_model_score}")

            # Clone and fit the best model
            best_model = clone(models[best_model_name])
            best_model.fit(x_train, y_train)

            # Predictions and metrics
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # MLflow tracking
            self.track_mlflow(best_model, classification_train_metric, best_model_name)
            self.track_mlflow(best_model, classification_test_metric, best_model_name)

            # Save final model
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
