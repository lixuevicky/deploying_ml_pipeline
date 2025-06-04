# Model Card
Author: Jasmine Li
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model predicts whether the person the salary is more than $50k with some demographic info.

## Intended Use
It can be used to make rough prediction on a person's income level given age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country.

## Training Data
The data is downloaded from the Census Bureau. Training data is 80% of all data.
Categorical features has been one-hot encoded.

## Evaluation Data
Evaluation data is 20% of all data.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
On test data, the performance is: precision: 0.74, recall: 0.63, f1_score: 0.68.

## Ethical Considerations
Since gender and race information are included, there can be some potential biases in the model. We can further check with Aequitas.

## Caveats and Recommendations
We can further explore the important features. 
Since this model is possibly controversial when we release it to public, we need to increase the explainability.
