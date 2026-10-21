# Decision Tree Algorithm --- Machine Learning Practice

## About This Repository
This repository documents my hands-on practice and learning of the
**Decision Tree algorithm** in Machine Learning using Python and
Scikit-learn.

The main goal is to understand how Decision Tree models work, train a
classifier on a real-world dataset, experiment with different model
settings, and evaluate its predictions using standard classification
metrics.
This is a learning and experimentation repository, not a
production-ready prediction system.


## Topics Practised

-   Understanding the Decision Tree classification algorithm
-   Selecting features and target variables
-   Encoding categorical features using one-hot encoding
-   Splitting data into training, validation, and testing sets
-   Training a Decision Tree classifier
-   Experimenting with different tree depths
-   Understanding class imbalance and balanced class weights
-   Exploring probability thresholds
-   Evaluating models using accuracy, precision, recall, and F1-score
-   Interpreting confusion matrices and classification reports
-   Analysing feature importance


## Tools and Technologies
-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Jupyter Notebook

## Practice Dataset
For the current practice, I used the IBM Telco Customer Churn dataset to
apply Decision Tree classification to a real-world-style classification
problem.

The selected input features were: - `tenure` - `MonthlyCharges` -
`Contract`

The target variable was `Churn`.

The dataset serves as a practical example for learning and experimenting
with the algorithm. The repository is not intended to represent a
complete customer churn solution.


## Experiments and Observations

During practice, I explored:

-   How changing `max_depth` affects model behaviour
-   How balanced class weights affect predictions for minority classes
-   How changing the classification threshold affects precision and
    recall
-   How to interpret feature importance
-   Why accuracy alone may not be sufficient for evaluating a
    classification model

These experiments helped me understand the trade-offs involved in
classification and the importance of evaluating models using multiple
metrics.


## Current Status

**Status: Algorithm practice completed; further improvements planned.**

This repository currently focuses on the machine learning workflow and
algorithm-level experimentation.

It does not currently include:

-   A backend application or REST API
-   A frontend interface
-   Model serving or deployment
-   Production-level monitoring

These may be explored in future projects as I continue learning machine
learning and backend development.


## Future Learning Goals

-   Practise additional machine learning algorithms
-   Compare Decision Trees with Random Forest and Logistic Regression
-   Experiment with more features and model evaluation techniques
-   Learn how to save and load trained models
-   Integrate a trained model with a backend API
-   Explore deployment when the model and application are ready


## Learning Outcome
Through this practice, I gained hands-on experience with the Decision
Tree classification workflow, model evaluation, class imbalance,
threshold selection, and feature importance using Scikit-learn.

This repository represents one step in my ongoing Machine Learning
learning journey.

------------------------------------------------------------------------

## **Note:** This is an educational practice repository. The experiments are intended for learning and may require further improvements before any real-world use.



# Random Forest Algorithm --- Machine Learning Practice

## About This Practice
After practising the Decision Tree algorithm, I continued in the same
repository with the **Random Forest classification algorithm** using
Python and Scikit-learn.

The purpose was to understand how an ensemble of decision trees works,
compare model behaviour, investigate overfitting, handle class
imbalance, tune hyperparameters, and explore classification thresholds.

This is a learning and experimentation exercise, not a production-ready
customer churn prediction system.


## Dataset and Features
I reused the IBM Telco Customer Churn dataset used in the Decision Tree
practice.

-   Target variable: `Churn`
-   Selected input features:
    -   `tenure`
    -   `MonthlyCharges`
    -   `Contract`

The `customerID` column was removed because it is an identifier. The
categorical `Contract` feature was converted into numerical columns
using one-hot encoding.

The resulting model input columns were:

-   `tenure`
-   `MonthlyCharges`
-   `Contract_Month-to-month`
-   `Contract_One year`
-   `Contract_Two year`

The target labels remained `No` and `Yes`, where `Yes` represents
customer churn.

The data was split into training and test sets using an 80/20 split with
`random_state=42` and `stratify=y`.

## Random Forest Workflow

The practice covered the following steps:
1.  Selected the input features and target variable.
2.  Removed the customer identifier.
3.  Encoded the categorical `Contract` feature using one-hot encoding.
4.  Split the data into training and test sets.
5.  Trained a baseline `RandomForestClassifier`.
6.  Evaluated predictions using accuracy, confusion matrix, precision,
    recall, and F1-score.
7.  Inspected feature importance.
8.  Compared training and test accuracy to understand overfitting.
9.  Restricted tree depth to study generalization.
10. Used balanced class weights to improve detection of the minority
    `Yes` class.
11. Compared model configurations using precision, recall, and F1-score.
12. Used `GridSearchCV` with five-fold cross-validation to tune
    hyperparameters.
13. Explored probability thresholds using a validation split.

## Baseline Random Forest

The initial model used 100 trees and `random_state=42`.

### Results
-   Training accuracy: **98.19%**
-   Test accuracy: **75.44%**
-   Precision for `Yes`: **54.5%**
-   Recall for `Yes`: **45.5%**
-   F1-score for `Yes`: **0.496**

Confusion matrix, with rows representing actual labels and columns
representing predicted labels in the order `No`, `Yes`:

``` text
[[893 142]
 [204 170]]
```

The model correctly identified many customers who stayed, but it missed
a substantial number of customers who actually churned. The difference
between training accuracy and test accuracy also indicated overfitting.

## Feature Importance
The baseline Random Forest reported the following feature importance
scores:

  Feature                       Importance
  --------------------------- ------------
  `MonthlyCharges`                0.606718
  `tenure`                        0.254682
  `Contract_Month-to-month`       0.101456
  `Contract_Two year`             0.028079
  `Contract_One year`             0.009066

In this model, `MonthlyCharges` had the highest split-based feature
importance, followed by `tenure` and `Contract_Month-to-month`.

These values describe how the model used features when making splits.
They do **not** prove that a feature causes churn, and they should not
be interpreted as the percentage of customers who churn because of that
feature.

## Reducing Overfitting with `max_depth`

The original model had a large training--test accuracy gap:

-   Training accuracy: **98.19%**
-   Test accuracy: **75.44%**

I then trained a Random Forest with `max_depth=5`.

### Depth-Limited Model Results
-   Training accuracy: **79.62%**
-   Test accuracy: **78.42%**
-   Precision for `Yes`: **63.9%**
-   Recall for `Yes`: **43.0%**
-   F1-score for `Yes`: **0.514**

Confusion matrix:

``` text
[[944  91]
 [213 161]]
```

Restricting tree depth reduced the training--test accuracy gap and
improved test accuracy in this experiment. However, recall for the churn
class remained relatively low. This showed that improving overall
accuracy does not necessarily mean detecting more churners.

## Handling Class Imbalance
The dataset contains more `No` labels than `Yes` labels. To give the
classes more balanced influence during training, I used:

``` python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)
```

### Balanced Random Forest Results
-   Test accuracy: **73.67%**
-   Precision for `Yes`: **50.3%**
-   Recall for `Yes`: **80.5%**
-   F1-score for `Yes`: **0.619**

Confusion matrix:

``` text
[[737 298]
 [ 73 301]]
```

The balanced model detected a much larger proportion of actual churners
than the depth-limited model, but it also produced more false-positive
predictions. This demonstrated the trade-off between detecting churners
and avoiding false alarms.

## Comparing the Main Model Configurations
  --------------------------------------------------------------------------------
  Model                 Test accuracy      Precision Recall (`Yes`)     F1 (`Yes`)
                                             (`Yes`)                
  -------------------- -------------- -------------- -------------- --------------
  Original Random              75.44%          0.545          0.455          0.496
  Forest                                                            

  Depth-Limited Random         78.42%          0.639          0.430          0.514
  Forest                                                            

  Balanced Random              73.67%          0.503          0.805          0.619
  Forest                                                            

  GridSearchCV-Tuned           73.46%          0.500          0.807          0.619
  Random Forest                                                     
  --------------------------------------------------------------------------------

The depth-limited model achieved the highest accuracy among these
configurations. The balanced and tuned models achieved substantially
higher recall for `Yes` and the highest F1-score, rounded to three
decimal places.

There is no universally best configuration without defining the business
objective. Accuracy, precision, recall, and F1-score measure different
aspects of performance.

## Hyperparameter Tuning with `GridSearchCV`

I used five-fold cross-validation to compare 24 combinations of Random
Forest settings.

The parameter grid was:

``` python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5],
    "class_weight": [None, "balanced"]
}
```

The model was tuned using F1-score for the positive `Yes` class.

### Best Parameters Found

``` python
{
    "class_weight": "balanced",
    "max_depth": 5,
    "min_samples_leaf": 1,
    "n_estimators": 200
}
```

-   Best mean cross-validation F1-score for `Yes`: **0.6156**
-   Parameter combinations evaluated: **24**
-   Cross-validation folds per combination: **5**
-   Total fits: **120**

The selected model used 200 trees, a maximum tree depth of 5, a minimum
leaf size of 1, and balanced class weights.

The cross-validation F1-score is an average over validation folds during
model selection; it is not test accuracy.

### Tuned Model Test Results
-   Test accuracy: **73.46%**
-   Precision for `Yes`: **50.0%**
-   Recall for `Yes`: **80.7%**
-   F1-score for `Yes`: **0.619**

Confusion matrix:

``` text
[[733 302]
 [ 72 302]]
```

The tuned model's results were very close to those of the manually
configured balanced Random Forest. Hyperparameter tuning did not produce
a substantial improvement in the previously used test-set results.

## Probability and Threshold Tuning
Random Forest can return class probabilities using `predict_proba()`.
The predicted probability for the `Yes` class can be compared with a
threshold to decide whether to predict `Yes` or `No`.

For example, with a threshold of `0.5`:

-   Probability below `0.5` → predict `No`
-   Probability equal to or above `0.5` → predict `Yes`

I created a fitting/validation split from the training data and trained
a validation model using the best parameters found by `GridSearchCV`. I
then compared thresholds of `0.3`, `0.4`, `0.5`, and `0.6` using
validation precision, recall, and F1-score for `Yes`.

### Validation Results
    Threshold   Precision (`Yes`)   Recall (`Yes`)   F1-score (`Yes`)
  ----------- ------------------- ---------------- ------------------
          0.3               0.401            0.923              0.559
          0.4               0.448            0.873              0.593
          0.5               0.504            0.803              0.619
          0.6               0.544            0.726              0.622

Among the four tested thresholds, `0.6` had the highest validation
F1-score. The lower threshold of `0.3` achieved the highest recall but
lower precision, meaning more actual churners were identified at the
cost of more false alarms.

Threshold `0.6` is only the best among the thresholds tested, not
necessarily the globally optimal threshold. The final choice should
depend on the cost of missed churners versus unnecessary
customer-retention actions.

## Important Evaluation Limitation

During this learning exercise, I used the test set for multiple model
comparisons. Therefore, the reported test results are useful for
practice, but they should not be treated as a completely unbiased final
performance estimate.

For a more rigorous evaluation, model and threshold selection should be
completed using training and validation data, followed by one final
evaluation on an untouched test set. The validation subset used for
threshold tuning was also part of the earlier training data used during
GridSearchCV, so a stricter workflow would reserve validation data
before hyperparameter selection.

## Random Forest Learning Outcomes
Through this practice, I gained hands-on experience with:

-   Training and evaluating a Random Forest classifier.
-   Understanding how an ensemble of decision trees makes predictions.
-   Interpreting feature importance.
-   Recognizing signs of overfitting and controlling tree depth.
-   Handling class imbalance with `class_weight="balanced"`.
-   Comparing accuracy, precision, recall, and F1-score.
-   Tuning hyperparameters using `GridSearchCV` and cross-validation.
-   Using `predict_proba()` and comparing classification thresholds.
-   Understanding trade-offs between missed churners and false-positive
    predictions.

## Current Status

**Status: Random Forest algorithm practice completed; further evaluation
and portfolio improvements are optional next steps.**

This repository documents algorithm practice and model evaluation. It
does not currently include a backend application, REST API, frontend,
model-serving system, or production monitoring.

The model uses only three original features and is intended for learning
rather than deployment. Additional feature engineering, a stricter final
evaluation, and application development would be needed before
considering real-world use.

## Tools and Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Jupyter Notebook

------------------------------------------------------------------------

# Overall Repository Learning Summary
This repository contains hands-on practice with **Decision Tree and
Random Forest classification** using the IBM Telco Customer Churn
dataset.

The Decision Tree practice introduced tree-based classification and
experiments with model settings. The Random Forest practice extended
that work to an ensemble of trees and included experiments with
overfitting, class imbalance, hyperparameter tuning, feature importance,
and probability thresholds.

The repository represents an ongoing Machine Learning learning journey.
The experiments are educational and are not intended to be a
production-ready customer churn solution.
