import os
import sys
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save the provided object to a file in binary format.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate classification models using GridSearchCV and return evaluation metrics.
    
    Returns a report with accuracy, precision, recall, and F1-score.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

    
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted')
            test_recall = recall_score(y_test, y_test_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            
            report[list(models.keys())[i]] = {
                "Test Accuracy": test_accuracy,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1 Score": test_f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file in binary format.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
