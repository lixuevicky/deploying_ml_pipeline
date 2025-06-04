import os
import sys
import logging
import pandas as pd
import pytest
import joblib

from data import process_data
from model import train_model, compute_model_metrics, inference

logging.basicConfig(
    filename='./log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# Add the necessary imports for the starter code.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

@pytest.fixture(scope="module")
def data():
    """
    Extract the data
    """
    data_path = os.path.join(file_dir, 'census.csv')
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def processed_X(data):
    """
    Extract pre-processed X
    """
    encoder = joblib.load(os.path.join(file_dir, 'encoder.pkl'))
    X, _, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder
    )
    return X


def test_process_data(data):
    """
    Check processed data has the same length with the input
    """
    try:
        X, y, _, _ = process_data(data, categorical_features=CAT_FEATURES, label="salary", training=True)
        assert len(X) == len(data)
        assert len(y) == len(data)
        logging.info("Testing processed data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing processed data: output length does not match input")
        raise err

def test_is_model():
    """
    Check saved model is present
    """
    try:
        model = joblib.load(os.path.join(file_dir, 'lrc_model.pkl'))
        logging.info("Testing model exists: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing model exists: The model wasn't found")
        raise err

def test_inference(processed_X):
    """
    Check predicted output length
    """
    try:
        model = joblib.load(os.path.join(file_dir, 'lrc_model.pkl'))
        assert len(inference(model, processed_X)) == len(processed_X)
        logging.info("Testing inference: SUCCESS")
    except AssertionError as err:
        logging.error("Testing inference: output length does not match input")
        raise err
   
