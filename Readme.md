# Decision Tree Algorithm — Machine Learning Practice

## About This Repository

This repository documents my hands-on practice and learning of the **Decision Tree algorithm** in Machine Learning using Python and Scikit-learn.

The main goal is to understand how Decision Tree models work, train a classifier on a real-world dataset, experiment with different model settings, and evaluate its predictions using standard classification metrics.

This is a learning and experimentation repository, not a production-ready prediction system.


## Topics Practised

- Understanding the Decision Tree classification algorithm
- Selecting features and target variables
- Encoding categorical features using one-hot encoding
- Splitting data into training, validation, and testing sets
- Training a Decision Tree classifier
- Experimenting with different tree depths
- Understanding class imbalance and balanced class weights
- Exploring probability thresholds
- Evaluating models using accuracy, precision, recall, and F1-score
- Interpreting confusion matrices and classification reports
- Analysing feature importance


## Tools and Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook


## Practice Dataset
For the current practice, I used the IBM Telco Customer Churn dataset to apply Decision Tree classification to a real-world-style classification problem.

The selected input features were:
- `tenure`
- `MonthlyCharges`
- `Contract`

The target variable was `Churn`.


The dataset serves as a practical example for learning and experimenting with the algorithm. The repository is not intended to represent a complete customer churn solution.


## Experiments and Observations
During practice, I explored:

- How changing `max_depth` affects model behaviour
- How balanced class weights affect predictions for minority classes
- How changing the classification threshold affects precision and recall
- How to interpret feature importance
- Why accuracy alone may not be sufficient for evaluating a classification model

These experiments helped me understand the trade-offs involved in classification and the importance of evaluating models using multiple metrics.


## Current Status
**Status: Algorithm practice completed; further improvements planned.**

This repository currently focuses on the machine learning workflow and algorithm-level experimentation.

It does not currently include:

- A backend application or REST API
- A frontend interface
- Model serving or deployment
- Production-level monitoring

These may be explored in future projects as I continue learning machine learning and backend development.


## Future Learning Goals
- Practise additional machine learning algorithms
- Compare Decision Trees with Random Forest and Logistic Regression
- Experiment with more features and model evaluation techniques
- Learn how to save and load trained models
- Integrate a trained model with a backend API
- Explore deployment when the model and application are ready

## Learning Outcome

Through this practice, I gained hands-on experience with the Decision Tree classification workflow, model evaluation, class imbalance, threshold selection, and feature importance using Scikit-learn.

This repository represents one step in my ongoing Machine Learning learning journey.

---

**Note:** This is an educational practice repository. The experiments are intended for learning and may require further improvements before any real-world use.
