# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from data import process_data
from model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv('census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, 'rfc_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(lb, 'lb.pkl')

train_pred = inference(model, X_train)
test_pred = inference(model, X_test)
precision, recall, f_one = compute_model_metrics(y_test, test_pred)
print(f'On test data, the performance is: precision: {precision}, recall: {recall}, f_one: {f_one}')

# a function that outputs the performance of the model on slices of the data
slice_feature = 'education'
X, y, _, _ = process_data(
    data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
preds = inference(model, X)
for cat in data[slice_feature].unique():
    slice_index = data.index[data[slice_feature] == cat]
    print(cat, compute_model_metrics(y[slice_index], preds[slice_index]))



