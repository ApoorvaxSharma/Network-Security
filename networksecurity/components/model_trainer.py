import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib

import dagshub

dagshub.init(repo_owner='apoorvasharmaxo', repo_name='Network-Security', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/apoorvasharmaxo/Network-Security.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="apoorvasharmaxo"
os.environ["MLFLOW_TRACKING_PASSWORD"]="cddb2b959967d3732d7ebd973ab7cd196bc31838"



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self, best_model, classificationmetric):
        mlflow.set_registry_uri("https://dagshub.com/apoorvasharmaxo/Network-Security.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)

            # Proper MLflow model logging (creates MLmodel, conda.yaml, etc.)
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(best_model, "model", registered_model_name="BestModel")
            else:
                mlflow.sklearn.log_model(best_model, "model")



    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
        }
        params = {
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                'learning_rate': [.1, .01, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        # Evaluate all models
        model_report: dict = evaluate_models(
            X_train, y_train, x_test, y_test,
            models=models, param=params
        )

        # Pick best model based on test_r2
        best_model_name = max(model_report, key=lambda k: model_report[k]["test_r2"])
        best_model_score = model_report[best_model_name]["test_r2"]

        best_model = models[best_model_name]

        # Predictions and metrics
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # ---- MLflow tracking ----
        from urllib.parse import urlparse
        import mlflow
        import mlflow.sklearn

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log metrics
            mlflow.log_metric("train_f1", classification_train_metric.f1_score)
            mlflow.log_metric("train_precision", classification_train_metric.precision_score)
            mlflow.log_metric("train_recall", classification_train_metric.recall_score)

            mlflow.log_metric("test_f1", classification_test_metric.f1_score)
            mlflow.log_metric("test_precision", classification_test_metric.precision_score)
            mlflow.log_metric("test_recall", classification_test_metric.recall_score)

            # Log model properly (creates MLmodel, conda.yaml, etc.)
            run_id = mlflow.active_run().info.run_id
            model_dir = f"saved_model_{run_id}"
            
            mlflow.sklearn.save_model(best_model,"saved_model")
            mlflow.log_artifacts("saved_model", artifact_path="model")
        # --------------------------

        # Save preprocessor + model locally
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        save_object("final_model/model.pkl", best_model)

        # Artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)