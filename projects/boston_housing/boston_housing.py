
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Machine-Learning-Engineer-Nanodegree" data-toc-modified-id="Machine-Learning-Engineer-Nanodegree-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Machine Learning Engineer Nanodegree</a></div><div class="lev2 toc-item"><a href="#Model-Evaluation-&amp;-Validation" data-toc-modified-id="Model-Evaluation-&amp;-Validation-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Model Evaluation &amp; Validation</a></div><div class="lev2 toc-item"><a href="#Project:-Predicting-Boston-Housing-Prices" data-toc-modified-id="Project:-Predicting-Boston-Housing-Prices-12"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Project: Predicting Boston Housing Prices</a></div><div class="lev2 toc-item"><a href="#Getting-Started" data-toc-modified-id="Getting-Started-13"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Getting Started</a></div><div class="lev2 toc-item"><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-14"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Data Exploration</a></div><div class="lev3 toc-item"><a href="#Implementation:-Calculate-Statistics" data-toc-modified-id="Implementation:-Calculate-Statistics-141"><span class="toc-item-num">1.4.1&nbsp;&nbsp;</span>Implementation: Calculate Statistics</a></div><div class="lev3 toc-item"><a href="#Question-1---Feature-Observation" data-toc-modified-id="Question-1---Feature-Observation-142"><span class="toc-item-num">1.4.2&nbsp;&nbsp;</span>Question 1 - Feature Observation</a></div><div class="lev2 toc-item"><a href="#Developing-a-Model" data-toc-modified-id="Developing-a-Model-15"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Developing a Model</a></div><div class="lev3 toc-item"><a href="#Implementation:-Define-a-Performance-Metric" data-toc-modified-id="Implementation:-Define-a-Performance-Metric-151"><span class="toc-item-num">1.5.1&nbsp;&nbsp;</span>Implementation: Define a Performance Metric</a></div><div class="lev3 toc-item"><a href="#Question-2---Goodness-of-Fit" data-toc-modified-id="Question-2---Goodness-of-Fit-152"><span class="toc-item-num">1.5.2&nbsp;&nbsp;</span>Question 2 - Goodness of Fit</a></div><div class="lev3 toc-item"><a href="#Implementation:-Shuffle-and-Split-Data" data-toc-modified-id="Implementation:-Shuffle-and-Split-Data-153"><span class="toc-item-num">1.5.3&nbsp;&nbsp;</span>Implementation: Shuffle and Split Data</a></div><div class="lev3 toc-item"><a href="#Question-3---Training-and-Testing" data-toc-modified-id="Question-3---Training-and-Testing-154"><span class="toc-item-num">1.5.4&nbsp;&nbsp;</span>Question 3 - Training and Testing</a></div><div class="lev2 toc-item"><a href="#Analyzing-Model-Performance" data-toc-modified-id="Analyzing-Model-Performance-16"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Analyzing Model Performance</a></div><div class="lev3 toc-item"><a href="#Learning-Curves" data-toc-modified-id="Learning-Curves-161"><span class="toc-item-num">1.6.1&nbsp;&nbsp;</span>Learning Curves</a></div><div class="lev3 toc-item"><a href="#Question-4---Learning-the-Data" data-toc-modified-id="Question-4---Learning-the-Data-162"><span class="toc-item-num">1.6.2&nbsp;&nbsp;</span>Question 4 - Learning the Data</a></div><div class="lev3 toc-item"><a href="#Complexity-Curves" data-toc-modified-id="Complexity-Curves-163"><span class="toc-item-num">1.6.3&nbsp;&nbsp;</span>Complexity Curves</a></div><div class="lev3 toc-item"><a href="#Question-5---Bias-Variance-Tradeoff" data-toc-modified-id="Question-5---Bias-Variance-Tradeoff-164"><span class="toc-item-num">1.6.4&nbsp;&nbsp;</span>Question 5 - Bias-Variance Tradeoff</a></div><div class="lev3 toc-item"><a href="#Question-6---Best-Guess-Optimal-Model" data-toc-modified-id="Question-6---Best-Guess-Optimal-Model-165"><span class="toc-item-num">1.6.5&nbsp;&nbsp;</span>Question 6 - Best-Guess Optimal Model</a></div><div class="lev2 toc-item"><a href="#Evaluating-Model-Performance" data-toc-modified-id="Evaluating-Model-Performance-17"><span class="toc-item-num">1.7&nbsp;&nbsp;</span>Evaluating Model Performance</a></div><div class="lev3 toc-item"><a href="#Question-7---Grid-Search" data-toc-modified-id="Question-7---Grid-Search-171"><span class="toc-item-num">1.7.1&nbsp;&nbsp;</span>Question 7 - Grid Search</a></div><div class="lev3 toc-item"><a href="#Question-8---Cross-Validation" data-toc-modified-id="Question-8---Cross-Validation-172"><span class="toc-item-num">1.7.2&nbsp;&nbsp;</span>Question 8 - Cross-Validation</a></div><div class="lev3 toc-item"><a href="#Implementation:-Fitting-a-Model" data-toc-modified-id="Implementation:-Fitting-a-Model-173"><span class="toc-item-num">1.7.3&nbsp;&nbsp;</span>Implementation: Fitting a Model</a></div><div class="lev3 toc-item"><a href="#Making-Predictions" data-toc-modified-id="Making-Predictions-174"><span class="toc-item-num">1.7.4&nbsp;&nbsp;</span>Making Predictions</a></div><div class="lev3 toc-item"><a href="#Question-9---Optimal-Model" data-toc-modified-id="Question-9---Optimal-Model-175"><span class="toc-item-num">1.7.5&nbsp;&nbsp;</span>Question 9 - Optimal Model</a></div><div class="lev3 toc-item"><a href="#Question-10---Predicting-Selling-Prices" data-toc-modified-id="Question-10---Predicting-Selling-Prices-176"><span class="toc-item-num">1.7.6&nbsp;&nbsp;</span>Question 10 - Predicting Selling Prices</a></div><div class="lev3 toc-item"><a href="#Sensitivity" data-toc-modified-id="Sensitivity-177"><span class="toc-item-num">1.7.7&nbsp;&nbsp;</span>Sensitivity</a></div><div class="lev3 toc-item"><a href="#Question-11---Applicability" data-toc-modified-id="Question-11---Applicability-178"><span class="toc-item-num">1.7.8&nbsp;&nbsp;</span>Question 11 - Applicability</a></div>

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[13]:

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# _Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each._  
# **Hint:** Would you expect a home that has an `'RM'` value of 6 be worth more or less than a home that has an `'RM'` value of 7?

# **Answer: **
# 
# _RM_: An increase in the average number of rooms among homes in the neighborhood should increase the value 'MEDV'. This is because homes with more rooms are typically larger and allow more activities to take place simultaneously.
# 
# _LSTAT_: An increase in the percentage of homeowner's considered working class should decrease the value of 'MEDV'. This is because homeowners that are less well off have less money to spend on housing.
# 
# _PTRATIO_: An increase in the ratio of students to teachers in primary and secondary schools in the neighborhood should decrease the value of MEDV. A low student to teacher ratio is typically seen as desirable since it means that each student will be able to receive more teacher attention and time. This means that parents (or potential parents) will be less willing to spend money to buy a house in a neighborhood with a high student to teacher ratio.
# 

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[3]:

from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score

#Don't need sklearn to calculate r squared
def performance_metricv1(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = 1 - sum(np.square(np.subtract(y_true, y_predict))) / sum(np.square(np.subtract(y_true, np.mean(y_true)))) 
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# *Would you consider this model to have successfully captured the variation of the target variable? Why or why not?* 
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[4]:

# Calculate the performance of this model
scorealt = performance_metricv1([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
scorealt2 = performance_metric([3, -0.5, 2, 7, 4.2], [3.14, 3.14, 3.14, 3.14, 3.14])
print("Using performance_metric, the model has a coefficient of determination, R^2, of {:.3f}.".format(score))
print("Using performance_metricv1, the model has a coefficient of determination, R^2, of {:.3f}.".format(scorealt))
print("Using just the mean, the model has a coefficient of determination, R^2, of {:.3f}.".format(scorealt2))


# **Answer:**
# Yes, because an R<sup>2</sup> of 0.923 indicates that 92.3% of the variation in the true value is explained by the predicted values. Compare this to an R<sup>2</sup> of 0 if we had predicted the mean of 3.14 for each point. However the interpretation of the coeffecient of determination very much depends on the context. Without any context to this model, it is hard to really say what the R^2 means. In some contexts, an R<sup>2</sup> of over 0.30 might be really good, however in others anything less than 0.99 might be unacceptable.

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[5]:

from sklearn.cross_validation import train_test_split

# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size = 0.2, random_state = 1234)

# Success
print("Training and testing split was successful.")


# ### Question 3 - Training and Testing
# *What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?*  
# **Hint:** What could go wrong with not having a way to test your model?

# **Answer: **
# The benefit to splitting a dataset into training and testing subsets for a learning algorithm is that it allows one to estimate the performance of the algorithm on unseen data. It is the performance on unseen data that we are usually interested in. We are almost never interested in making predictions on data for which we already observed the outcomes variables. For example, with this data set we might want to predict how changes in neighborhood conditions will influence the average selling price. Therefore we are interested in predicting the outcome variables for particular combinations of features that are not included in the training set. A model that only performs well on the training set and its particular combinations of features and outcomes is not very interesting or useful. Instead, we want a model that generalizes well to unobserved conditions. By putting some of the data aside as a testing set, and not fiting the model to this data, we can then estimate the performance of the model on unobserved data. We will want to construct a model that performs well on unseen data, rather than a model that performs well on the observed data. So the goal will be to construct a model that performs well on the test set, rather than on the training set.

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[6]:

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# *Choose one of the graphs above and state the maximum depth for the model. What happens to the score of the training curve as more training points are added? What about the testing curve? Would having more training points benefit the model?*  
# **Hint:** Are the learning curves converging to particular scores?

# **Answer: **
# I choose the bottom left chart with a maximum depth of 10. As more training points are added, it seems that the score of the training curve is decreasing ever so slightly. It seems that the testing score is stable as more training points are added, after increasing with the first 200 training points. Based on the other graphs, it is likely that adding more training points would benefit the model, as more training points could help the training and test scores converge to a score that is higher than the current maximum test score. However based on the rate of convergence, this might require an infeasibly large number of training points and an improvement in the test score is not guaranteed. It is possible that a maximum depth of 10  simply produces a model that is too complex for the data set given that we only have three features.

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# Run the code cell below and use this graph to answer the following two questions.

# In[7]:

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# *When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?*  
# **Hint:** How do you know when a model is suffering from high bias or high variance?

# **Answer: **
# When the model is trained with a maximum depth of 1, the model suffers from high bias. This can be observed from the fact that both the training score is quite low, but is quite similar to the validation score. A low score indicates a high bias, and the similarity of the validation and train scores indicate a low variance.
# 
# When the model is trained with maximum depth of 10, the model suffers from high variance. This can be observed from the fact that the training and validation scores are quite far apart, but the training score is high. The fact that they are far apart indicates a high variance, but a high train score indicates a low bias.

# ### Question 6 - Best-Guess Optimal Model
# *Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?*

# **Answer: **
# 
# It seems that a maximum depth of 4 generalizes best to unseen data. I say this because a maximum depth of 4 produces the highest validation score. The validation score is essentially an estimate of the performance of the model on unseen data, since the validation score was calculated using data that the model did not use.

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# *What is the grid search technique and how it can be applied to optimize a learning algorithm?*

# **Answer: **
# A grid search is the repeated training of a certain algorithm with different values for various "hyperparameters". The relative performance of these different hyperparameter values can then be evaluated on a holdout validation data set or with cross fold validation. The model and its associated hyparameters that produced the lowest validation error can then used as the final model, since the grid search has shown that these are the best hyperparameters to use. Hyperparameters are parameters that an algorithm does not automatically learn from the data, and must be set by the machine learning engineer.

# ### Question 8 - Cross-Validation
# *What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?*  
# **Hint:** Much like the reasoning behind having a testing set, what could go wrong with using grid search without a cross-validated set?

# **Answer: **
# The k-fold cross-validation training technique is when the out of sample performance of a model is estimated by dividing the training data into k folds (usually a number between 5 and 10). The model is fit iteratively k times, each time on all of the data except for the training points that lie in the kth fold. The data points in the kth fold are used to estimate the error of the kth model. The validation error is calculated as the average of the error from these k validation folds. The benefit to the k-fold validation compared to using a holdout set is that all of the data ends up being used for both training and validation. This is beneficial when one does not have too many data points to begin with. When one has many data points already, k-fold validation could take a lot longer compared to using a hold out set. It is necessary to use a hold out validation set or k-fold validation when doing a grid search since otherwise one would not be able to properly evaluate the different values of hyperparameters. Evaluating the hyperparameters using the training score is inappropiate since hyperparameters that produce a more complex model will always lead to a lower training loss. We have already seen that a lower training loss is not necessarily associated with a lower error on unseen data. 

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# Please note that ShuffleSplit has different parameters in scikit-learn versions 0.17 and 0.18.
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[10]:

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'

from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    # sklearn version 0.18: ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)
    # sklearn versiin 0.17: ShuffleSplit(n, n_iter=10, test_size=0.1, train_size=None, random_state=None)
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {"max_depth" : range(1,11)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# _What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?_  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[11]:

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))


# **Answer: **
# The optimal model has maximum depth of 4. This is the same as my guess for Question 6. This is unsurpising, since my answer from Queston 6 was based off of the chart produced by vs.ModelComplexity, which plotted the 10-fold validation score against the maximum depth. I simply read off the highest validation score from that chart.

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# *What price would you recommend each client sell his/her home at? Do these prices seem reasonable given the values for the respective features?*  
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[36]:

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))


# **Answer: **
# Based on the above estimated values, I would recommend that:
# 
# * Client 2 sell at 250,000. It is reasonable to recommend the lowest listing price to Client 2 because they have the smallest house (as measured by the number of rooms) in addition to high neighborhood poverty levels and student-teachers ratio which are undesirable.
# 
# * Client 3 sell at $1,000,000. It is reasonable that Client 3 should expect to receive the highest value for their house because they have the largest house and a low neighborhood povery level and a low student-teacher ratio in nearby schools is very desirable.
# 
# * Client 1 sell at $400,000. Client 1 should expect to receive a price for their house between that of Client 2 and 3. This is because the number of rooms in their house falls between that of Client 2 and Client 3. In addition, the desirabiliy of their neighborhood, as measured by neighborhood poverty level and student to teacher ratio, falls between that of Client 1 and Client 2.
# 
# Notice that I would recommend my clients list their house in multiples of $50,000 instead of the exact value predicted by the model. House prices are often listed as multiples of certain numbers such as 5,000 or 25,000. These round numbers are easier for people to read and interpet. The values provided by the model are not expected to reflect the actual selling price of any given house exactly. It would be strange for me to recommend the very precise numbers produced by the model to my clients. In real life I would probably look at multiple models and incorporate my personal experience when making a recommendation to my clients.

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with the data it's trained on.

# In[37]:

vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# *In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.*  
# **Hint:** Some questions to answering:
# - *How relevant today is data that was collected from 1978?*
# - *Are the features present in the data sufficient to describe a home?*
# - *Is the model robust enough to make consistent predictions?*
# - *Would data collected in an urban city like Boston be applicable in a rural city?*

# **Answer: **
# I can't forsee the model constructed from this data set being used in any real-world setting. First of all, the number of samples is quite low. This is partly because each data point corresponds to a neighborhood rather than and individual housing transaction. The relatively small size of the data set can also be attributed to the fact that it only contains data on Boston housing prices.
# 
# The fact that the data only contains data for Boston would also limit its ability to generalize. For example, in areas with less children, the student-teacher ratio could have less of an impact on housing prices. In addition, the fact that this data set comes from 1978 severely limits it applicability to modern day housing prices. I am certain that the definition of "working poor" is different than it was forty years ago.
# 
# Finally, the inclusion of only three explanatory features is quite limiting. I would expect a large multitude of features to be neccesary in order to make predictions that are reliable and accurate enough to be used in real world situations, such as the sale of a house worth several hundred thousand dollars. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
