**Credit Card Fraud Detection: Classical ML vs. Deep Learning**

This task distinguishes a classical ML approach from a deep learning approach to credit card fraud detection.

Comparative Study of Traditional Machine Learning versus Deep Learning Methods to Detect Financial Fraud in a Highly Imbalanced Dataset.


**Project Overview**

Financial Fraud Detection is a typical imbalanced classification problem: Fraudulent transactions account for less than 0.2% of the total number of transactions, so a naive model that always predicts "not fraud" is 99.8% accurate but utterly useless.

The goal of this project is to thoroughly compare the performance of 7 experiments between 2 model families: Classical ML and Deep Learning, and 3 class imbalance treatment strategies: None, SMOTE and Class Weights, to identify the most operationally viable approach for fraud detection.

Primary metric: Precision-Recall AUC (PR-AUC) - more informative than ROC-AUC on highly skewed datasets.
Repository Structure

credit-card-fraud-detection/
│
├── Shem_Ayioka_Summative.ipynb   ← Main notebook (all code + analysis)
├── README.md                      ← This file
└── requirements.txt               ← Python dependencies


Note: The dataset is downloaded automatically inside the notebook via gdown from Google Drive. No manual download is needed.


**Methodology**

Preprocessing


Time and Amount scaled using RobustScaler (resistant to outliers)
Preserved 80/20 split between train and test to maintain class balance.
Data leakage was avoided by applying SMOTE to just training data


**Imbalance Strategies Compared**

DescriptionNoneTrain on raw imbalanced data: SMOTE: Synthesize the minority class (fraud) to balance the data, with a heavier weight given to their loss function.

**Experiments**

Experiments

#ModelTreatment1Logistic RegressionNone2Random ForestNone3Random ForestSMOTE4MLP (Functional API)None5MLP (Functional API)SMOTE6MLP (Functional API)Class Weights7MLP Complex (Functional API)Class Weights

**Key Findings**


Random Forest (No Treatment) showed the highest PR-AUC and a good precision and recall balance, making it the most operationally viable model.
In the case of MLP with Class Weights, the Recall was extremely high, with almost all fraud being caught, but with a horrendous number of false positives — thousands of legitimate transactions being flagged as errors.
SMOTE + Deep Learning did not perform well due to generating synthetic samples on already abstracted PCA features which led to noisy decision boundaries for gradient-descent-based models.
On tabular and PCA transformed financial data sets, the tree-based ensembles beat deep learning, showing that deep learning is not necessarily better on structured, imbalanced sets.



**How to Run**

Google Colab is the recommended option.The first option is Google Colab (Recommended).


Open Shem_Ayioka_Summative.ipynb on Google Colab.
Run all cells, from top to bottom (Runtime → Run all)
The datasets are downloaded automatically, no setup is required.


Option 2: Local Jupyter

bash# 1. Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install dependencies
Install the requirements from requirements.txt file:

# 3. Launch Jupyter
jupyter notebook Shem_Ayioka_Summative.ipynb


**Requirements**

pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
gdown

Attach all in one operation:

pip install -r requirements.txt


**Reproducibility**

All random seeds are set at the top of the notebook:

pythonSEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

When the notebook is run from top to bottom, it will yield the same results each time.


**Limitations**


Loss of interpretability: V1–V28 are anonymized principal components, which do not allow any domain-specific feature engineering or interpretation of the model decision.
Static snapshot: The data is only for 2 days in 2013. Fraud patterns change, this model will experience concept drift in production.
The synthetic oversampling of the abstract principal components creates noisy and overlapping boundaries that are detrimental to neural networks, especially.



Author

_Shem Ayioka_

Summative Assessment — Machine Learning
