
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# In[1]:

##Software versions
import sys
import numpy
import pandas
import sklearn

print(sys.version)
print("Numpy version: %s" % numpy.__version__)
print("Pandas version: %s" % pandas.__version__)
print("Sklearn version: %s" % sklearn.__version__)


# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[2]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.
# 
# **Hint:** You may need to look at the table above to understand how the `'income'` entries are formatted. 

# In[3]:

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = sum(data.income == ">50K")

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = sum(data.income == "<=50K")

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k / n_records

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2%}".format(greater_percent))


# ----
# ## Preparing the Data
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.

# In[4]:

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 

# In[5]:

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

# In[6]:

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
#  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_raw'` data.
#  - Convert the target label `'income_raw'` to numerical entries.
#    - Set records with "<=50K" to `0` and records with ">50K" to `1`.

# In[7]:

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
income = (income_raw == ">50K").astype(int)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
print(encoded)


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.

# In[8]:

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Evaluating Model Performance
# In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 

# ### Question 1 - Naive Predictor Performace
# *If we chose a model that always predicted an individual made more than \$50,000, what would that model's accuracy and F-score be on this dataset?*  
# **Note:** You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.

# In[9]:

# TODO: Calculate accuracy
accuracy = greater_percent

# TODO: Calculate F-score using the formula above for beta = 0.5
fscore = (1 + 0.5**2) * (accuracy * 1) / ((0.5**2 * accuracy) + 1)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# ###  Supervised Learning Models
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# - *Describe one real-world application in industry where the model can be applied.* (You may need to do research for this — give references!)
# - *What are the strengths of the model; when does it perform well?*
# - *What are the weaknesses of the model; when does it perform poorly?*
# - *What makes this model a good candidate for the problem, given what you know about the data?*

# **Answer: **
# __Logistic Regression__
# 
# _Real word application_:
# One example of a real world application where this model can be applied is predicting the difficulty of parking at a given place and time. Google implemented this feature in their Google Maps products using logistic regression in February 2017.
# https://research.googleblog.com/2017/02/using-machine-learning-to-predict.html
# 
# 
# _Strengths of the model_:
# Logistic regression has several strengths. The first of these is that it is highly parametric. That means that it very easy to interpret the resulting model, in particular its parameters. This is very important when one has explain the model and justify the decisions one based on the model. This need can quite easily arise in a business setting, and the ability to explain a model to upper management, customers or regulators should not be underestimated..
# 
# The second strength of logistic regression is that it outputs unbiased probabilities. Several models such as Support Vector Machines or K Nearest Neighbours do not output probabilities without signifcant modifications. While other models such as Naive Bayes or decision trees also output probabilities, they are not unbiased and require adjustments before they can  The probabilities from the logistic regression algorithm can be used without further calibration. Reference: http://scikit-learn.org/stable/modules/calibration.html
# 
# A third strength of logistic regression is that it efficient to train the model and to make predictions using the resulting model. This is important when one has a large quantity of data and perhaps not much time. In particular, when one has a great deal of features compared to the number of training examples logistic regression can perform well.Logistic regression is a good baseline.
# 
# _Weakness of the model_:
# The biggest drawback of logistic regression stems from its simplcicity. It cannot deal automatically with non-linear relationships between the class and the explanatory variables. While it is possible to manually add non-linear features, this process can be time consuming. 
# 
# _Why is it a good candiate?_:
# Logistic regression is a good candidate for this problem, but not because of preexisting information that I have regarding the data set. Since there are only 13 (103 after one hot encoding) features, logistic regression is not needed to prevent over fitting. In fact, with so few predictor variables, it is expected that logistic regression will underperform compared to other models that can learn more complicated, non-linear, relationships. In addition, with less than 50 000 records, the data set is quite small. So logistic regression is not needed for performance reasons. However, I think that logistic regression is an appropiate model for this problem because it is good to include it as a baseline. If other, more complicated models, do not outperform it by a significant margin, a logistic regression model should be used since it's improved interpretability could represent a large business value to the charity.
# 
# 
# __Nearest neighbours__
# 
# _Real word application_:
# When browsing or otherwise accessing a piece of content on the internet, the nearest neighbours algorithm can be used to serve up an ad that is similar to the content that is being accessed. Facebook would like to use such a method to serve ads related to a video that a user is watching. The ability to engage in such behaviour is potentially very profitable, important enough that Facebook has developed a more efficient implementation of the algorithm. 
# 
# https://techcrunch.com/2017/03/29/similarity-search/
# 
# _Strengths of the model_:
# An important strength of the nearest neighbour algorithm is that it is fully non-parametric. Given enough training points, it can learn any relationship. Another strength is that one inspect the nearest neighours for each training point. While not as easy to interpret as logistic regression, this can help explain and justify the results of the model to others.
# 
# _Weakness of model_:
# The biggest weakness of the nearest neighbours algorithm is that it does not work well when the data set has many features. This is known as the curse of dimensionality. It means that as the number of features increases, the required number of training examples increases exponentially. As a result, there are many data sets to which the k nearest neighbours algorithm cannot be applied without reducing the number of dimensions. 
# 
# In addition to not scaling well with the number of feature dimensions, the k nearest neighbours algorithm can also be slow when making predictions. This is because the k nearest neighbours algorithm does not do any "pretraining". It does not learn a relationship between features and labels that is used at prediction time. At prediction time, the k-nearest algorithm looks anew at all the training points in order to produce a prediction for a testing sample.  
# 
# _Why is it a good candiate?_:
# K nearest neighbours is a good candidate for this problem since the with less than 50 000 records, the algorithm will not be too slow when making predictions. In addition, the number of features is not too high relative to the number of training examples, so we should not be encountering the curse of dimensionality. Finally, given that I have selected to test logistic regression, a strongly parametric model, it will interesting to compare and contrast its results with k nearest neighbours, a non-parametric model.
# 
# __Gradient boosted trees__
# 
# _Real word application_:
# One example of a real world application where boosted trees can be applied is predicting whether a deliquent mortgage loan will re-perform after receiving a modification. This would be a very important application because the ability to sucessfuly predict whether a loan will re-perform could lead a many opportunities to make a profit by buying the deliquent mortgages at a low price and then subsequently selling them for a higher price when they re-perform.
# 
# I obtained the idea for this application from the following report:
# http://us.milliman.com/uploadedFiles/insight/2017/enhanced-vision.pdf
# 
# _Strengths of the model_:
# The gradient boosted trees algorithm has many strengths. While the algorithm is not as parametric as logistic regresion, it can still produce indicators of the relative importance of each feature. This can be very helpful when quality assuring the model, although the feature importances can be hard to interpret. Unlike logistic regression, boosted trees can learn non-linear relationships. In addition, because the boosted trees ensembles weak learners, it is not likely to overfit to the data. This means that in many situations gradient boosted trees provide very good results.
# 
# Finally, another strength of boosted trees is that the model can output probabilities without additional processing. These probabilities can be used in a myriad of ways. For example, earlier I gave an example of using boosted trees to predict whether a mortgage loan will re-perform. In this situation, one could start to test the real world profitability of the predictions by only buying and modifiying mortgages that are predicted to re-perform with a high proability. If the high probability loans prove profitable, one can commence to purchase loans with a lower probability (but greater than 50%) of re-performance. Finally, the boosted trees alogrithm is usually quite efficient to train and predict. This is very helpful when quickly diving into a problem.
# 
# 
# _Weakness of the model_:
# Compared to simpler, more parametric models, such as logistic regression, boosted trees are a bit of a "black box". It is not easy to inspect or explain the predictions output by the boosted trees algorithm. This can be a significant drawback if there is need to justify why certain actions were taken based on the model. In addition, boosted trees also contain many more hyper-parameters compared to a simpler method such as logistic regression. These hyper-parameters might need to be manually tuned in order to achieve optimal results. However one can expect that boosted trees will outperform logistic regression with default hyper-parameter values, especially considering the limited number of features in this dataset.
# 
# _Why is it a good candiate?_:
# Given that we have only 13 features, the boosted trees method is a good candidate since better performance will be probably be obtained by using a model that can learn non-linear relationships. Especially considering that we have 40k+ records, there is a low risk of overfitting. Gradient boosted trees are a good compromise between a strongly parametric model such as logistic regression and a fully non-parametric model such a k nearest neighbours.

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# In[10]:

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train.iloc[:sample_size, :], y_train.iloc[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train.iloc[:300,:])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train.iloc[:300], predictions_train[:300])
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train.iloc[:300], predictions_train[:300], beta = 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

# In[11]:

# TODO: Import the three supervised learning models from sklearn


# TODO: Initialize the three models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# TODO: Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = LogisticRegression(random_state = 123)
clf_C = GradientBoostingClassifier(random_state = 123)


# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(0.01 * y_train.shape[0])
samples_10 = int(0.1 * y_train.shape[0])
samples_100 = y_train.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# *Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000.*  
# **Hint:** Your answer should include discussion of the metrics, prediction/training time, and the algorithm's suitability for the data.

# **Answer: **
# I believe that gradient boosted trees classifier is the most appropiate model for the task of identifying individuals that make more than $50,000. This is because the above results show that the boosted trees classifier performs best, as measured by both the accuracy and the F-score, of the three tested models. 
# 
# It is true that the boosted trees classifier has the longest training time, at about 10 seconds. However, 10 seconds is not very long. If we get more data, we could re-train the model and even then it should not take very long. Since a model is only trained once, the prediction time is more important. Here, the boosted trees perform very well. In fact, given the long prediction time of the k nearest neighbours algorithm, the prediction time of the boosted trees classifier does not even register.
# 

# ### Question 4 - Describing the Model in Layman's Terms
# *In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

# **Answer: ** 
# The gradient boosted trees classifier builds a series of decision trees. A decision tree splits the training examples into groups by looking at single feature at a time. It tries to make certain that the samples in these groups have the same label. Because the gradient boosted trees classifier builds a series of decision trees, each individual decision tree does not have to be very accurate. Instead, each tree tries to improve upon the trees that came before.  Prediction is accomplished by outputting the results of all of these weak decision trees and then summing together their outputs. This produces a probability of whether a particular record belongs to the target class.

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Initialize the classifier you've chosen and store it in `clf`.
#  - Set a `random_state` if one is available to the same state you set before.
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
#  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# 
# **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

# In[12]:

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = GradientBoostingClassifier(random_state = 123)

# TODO: Create the parameters list you wish to tune
parameters = {'learning_rate' : [0.01, 0.05, 0.1], 'n_estimators' : [100, 500, 1000], 'max_depth' : [3, 5, 8]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta = 0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring = scorer, n_jobs = 6)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# ### Question 5 - Final Model Evaluation
# _What is your optimized model's accuracy and F-score on the testing data? Are these scores better or worse than the unoptimized model? How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

# #### Results:
# 
# |     Metric     | Benchmark Predictor | Unoptimized Model | Optimized Model |
# | :------------: | :-----------------: | :---------------: | :-------------: | 
# | Accuracy Score | 0.2478              | 0.8630            | 0.8710          |
# | F-score        |    0.2917           |  0.7395           |    0.7530       |
# 

# **Answer: **
# 
# The optimized model is better than the unoptimized model. This makes sense since the unoptimized model, with the sklearn defaults, was included in the grid search (depth of 3, 100 estimators, learning rate of 0.1). However the optimized option only results in an improvement of 0.0135 in the F-score, with a similar improvement in the accuracy score. Both the unoptimized and the optimized model perform much better than the benchmark predictor. Using either the optimized or unoptimized boosted trees classifier should result in much better business decisions for the client charity than trying to solicit from all individuals under the assumption that they all make over $50,000.
# 

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data.  
# _Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?_

# **Answer:**
# 
# I believe that that the following variables are the most important, in order of importance:
# 
# 1.  __education-num__: Education-num is the number of years of education that an individual has completed. This an important pre-requisite for many high paying jobs. Even when two people have the same job title, one can expect the person with the greater amount of education to receive a higher salary.
# 2.  __occupation__: This variable indicates the job title of the individual. One can expect this to be an important predictor, since people with certain, more prestigous, job titles should earn more.
# 3.  __workclass__: This variable is an indicator of whether an individual is working or not, and for what kind of organization, if working. One can expect those individiuals that are working to have a higher income on average, and for those people working for the federal government to earn more.
# 4.  __age__: This seems to be the age of the individual, in years. This should be an important predictor since older, more experienced, inviduals should be compensated more for their experience. 
# 5.  __hours-per-week__: This variable is the average number of hours per week than an individual has worked. It would make sense for those people who work more to earn more.

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[13]:

# TODO: Import a supervised learning model that has 'feature_importances_'

# TODO: Train the supervised model on the training set 
model = GradientBoostingClassifier(random_state = 123, learning_rate = 0.05, n_estimators = 1000, max_depth = 3).fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# _How do these five features compare to the five features you discussed in **Question 6**? If you were close to the same answer, how does this visualization confirm your thoughts? If you were not close, why do you think these features are more relevant?_

# **Answer:**
# 
# Three of the features that I choose appear in the top 5. Age appears before hours-per-week, as I predicted. However of the five most predictive features, eductation-num is the least predictive. At first I was surprised, but then I realized two things. 
# 
# 
# First, there is another variable, "education_level", which seems to be measuring the same thing as education-num, but it is a categorial variable rather than a numeric variable. Since these two variables are highly correlated, it makes sense that neither one would appear to be particularly strong predictor on its own.
# 
# Secondly, I realized that Question 6 was ill posed. Question 6 asked me to select and explain what I thought would be the five most important predictors, out of 13 predictor variables. However, as one can clearly see from the "Implementation: Data Preprocessing" section, there are in fact 103 features after one-hot-encoding. This is because each categorial feature was split into several numeric variables using the pandas.get_dummies() function. Since every categorical variable has now been split into several numeric child variables, it is unlikely for one of these child variable to appear as on the top 5 predictors. In fact, it was impossible for the occupation and the work_class variables to appear in the above chart, since they do not exist in X_train. Only their children, variables with different names, exist.

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# In[14]:

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# ### Question 8 - Effects of Feature Selection
# *How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?*  
# *If training time was a factor, would you consider using the reduced data as your training set?*

# **Answer:**
# Using only five features, the model performs worse, especially as measured by the F-score. It obtains a F-score that is 0.0564 below the final model that uses all the features. This a disappointing result. However, if training time was a factor, for example if training time dropped from several days to a few hours when using less features, I would consider using the reduced data as a training set. However, given that it took less than 10 seconds to train the unoptimized model on the full training set, I would not reccomend using the reduced feature set in this case. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
