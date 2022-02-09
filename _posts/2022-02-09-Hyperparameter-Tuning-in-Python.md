---
layout: xxxx
title: Hyperparameter Tuning in Python
date: 2022-02-09
tags: datacamp, python, hyperparameter
comments: true
---

## Lesson 1 : Hyperparameters and Parameters


### Parameter
```python 
log_reg_clf = LogisticRegression() 
log_reg_clf.fit(X_train, y_train)
print(log_reg_clf.coef_)
```
![](https://i.imgur.com/sGvF0Ty.png)


```python

 # Get the original variable names
original_variables = list(X_train.columns)

 # Zip together the names and coefficients
zipped_together = list(zip(original_variables,log_reg_clf.coef_[0]))
coefs = [list(x) for x in zipped_together]

 # Put into a DataFrame with column labels
coefs = pd.DataFrame(coefs, columns=[“Variable”, “Coefficient”])

coefs.sort_values(by=["Coefficient"], axis=0, inplace=True, ascending=False)
print(coefs.head(3))
```

![](https://i.imgur.com/AMmS5yI.png =200x130)


* Random Forest

```python 
	# A simple random forest estimator
rf_clf = RandomForestClassifier(max_depth=2)
rf_clf.fit(X_train, y_train)
	# Pull out one tree from the forest 
chosen_tree = rf_clf.estimators_[7]
```

![](https://i.imgur.com/ehls9me.png)


```python
# Get the column it split on
split_column = chosen_tree.tree_.feature[1]
split_column_name = X_train.columns[split_column]
# Get the level it split on
split_value = chosen_tree.tree_.threshold[1]
print("This node split on feature {}, at a value of {}"
        .format(split_column_name, split_value))
```


* Example:
```python
# Create a list of original variable names from the training DataFrame
original_variables = list(X_train.columns)

# Extract the coefficients of the logistic regression estimator
model_coefficients = log_reg_clf.coef_[0]

# Create a dataframe of the variables and coefficients & print it out
coefficient_df = pd.DataFrame({“Variable” : original_variables, “Coefficient”: model_coefficients})
print(coefficient_df)

# Print out the top 3 positive variables
top_three_df = coefficient_df.sort_values(by=‘Coefficient’, axis=0, ascending=False)[0:3]
print(top_three_df)
```


* Visualize:
```python

# Extract the 7th (index 6) tree from the random forest
chosen_tree = rf_clf.estimators_[6]

# Visualize the graph using the provided image
imgplot = plt.imshow(tree_viz_image)
plt.show()

# Extract the parameters and level of the top (index 0) node
split_column = chosen_tree.tree_.feature[0]
split_column_name = X_train.columns[split_column]
split_value = chosen_tree.tree_.threshold[0]

# Print out the feature and level
print(“This node split on feature {}, at a value of {}”.format(split_column_name, split_value))

```
![](https://i.imgur.com/Teem9J2.png =500x300)


### Hyperparameter
> the algorithm does not learn these

* For random forest, some 
	* hyperparameter will not help model importance
		* n_jobs
		* random_state
		* verbose

	* some can
		* n_estimators 
		* max_features
		* max_depth & min_sample_leaf (important for overfitting)
		* criterion

* How to find hyper parameter that matters
	* academic papers
	* blogs and tutorials from trusted source
	* scikit-learn documentation
	* experience

* Silly hyper parameter values
	* random forest with low number of trees
	* 1 neighbor in KNN algorithm
> sensible values for hyper parameters 

* Learning curves
	* handy trick for generating values
		* range()
		* np.linspace(start, end, num)

* Example
```python

# Set the learning rates & accuracies list
learn_rates = np.linspace(0.01, 2, num=30)
accuracies = []

# Create the for loop
for learn_rate in learn_rates:
    # Create the model, predictions & save the accuracies as before
    model = GradientBoostingClassifier(learning_rate=learn_rate)
    predictions = model.fit(X_train, y_train).predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

# Plot results    
plt.plot(learn_rates, accuracies)
plt.gca().set(xlabel='learning_rate', ylabel='Accuracy', title='Accuracy for different learning_rates')
plt.show()

```

![](https://i.imgur.com/awBZqp1.png)



## Lesson 2 : Grid search
* Pros
* Cons :
	* Computation
	* un-informed method

* Example (for loop method)
```python
# Create the function
def gbm_grid_search(learning_rate, max_depth):

    # Create the model
    model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth)
    
    # Use the model to make predictions
    predictions = model.fit(X_train, y_train).predict(X_test)
    
    # Return the hyperparameters and score
    return([learning_rate, max_depth, accuracy_score(y_test, predictions)])


results_list = []

# Create the new list to test
subsample_list = [0.4, 0.6]

for learn_rate in learn_rate_list:
    for max_depth in max_depth_list:
    
        # Extend the for loop
        for subsample in subsample_list:
            
            # Extend the results to include the new hyperparameter
            results_list.append(gbm_grid_search_extended(learn_rate, max_depth, subsample))
            
# Print results
print(results_list)            

```


* GridSearchCV Object of sklearn
* Steps
	* algorithm
	* hyperparameter
	* range of hyper parameter
	* cross-validation scheme
	* score function to decide which is the best one

> CV
> 
![](https://i.imgur.com/ws17FTH.png)


* refit
	* fit the best hyper parameter to training data
	* handy option
	* to save the best estimator 

* Parallel computing - n_jobs
```python
import os
print(os.cpu_count())

```

* return_train_score

* Example
```python
# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2,4,8,15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)



```


### Analyze the output
* .property
	* cv_results_
		* time column
		* param_ column
		* test_score
		* rank_test_score
	* best_params_
	* best_score_
	* best_index_, the row in ‘cv_results_.rank_test_score’
	* best_estimator_
	* others
		* scorer
		* n_splits_
		* refit_time_

> pd.set_option(“display.max_colwidth”, -1)

* Example
```python

# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
print(cv_results_df)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, [“params”]]
print(column)

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df[“rank_test_score”] == 1]
print(best_row)

# Print out the ROC_AUC score from the best-performing square
best_score = grid_rf_class.best_score_
print(best_score)

# Create a variable from the row related to the best-performing square
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)
best_row = cv_results_df.loc[[grid_rf_class.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rf_class.best_params_[“n_estimators”]
print(best_n_estimators)


# See what type of object the best_estimator_ property is
print(type(grid_rf_class.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))

```

## Lesson 3: Random Search
* Why does this work?
> Bengio & Bergstra(2012): This paper shows empirically and theoretically that randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid
	* not every hyper parameter is as important
	* a little trick of probability

* A probability
	* assume there is 5% best models
	* trial 1: 0.95 to miss all the best models
		* trial 2 : 0.95*0.95 to miss all
			* trial n : 0.95^n to miss all
			* get the best models 1-(0.95^n)
			*  1-(0.95^n) > 0.95, n >=59

* Example :
```python
# Create a list of values for the learning_rate hyperparameter
learn_rate_list = list(np.linspace(0.01,1.5,200))

# Create a list of values for the min_samples_leaf hyperparameter
min_samples_list = list(range(10,41))

# Combination list
combinations_list = [list(x) for x in product(learn_rate_list, min_samples_list)]

# Sample hyperparameter combinations for a random search.
random_combinations_index = np.random.choice(range(0, len(combinations_list)), 250, replace=False)
combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]

# Print the result
print(combinations_random_chosen)

```


```python
# Create lists for criterion and max_features
criterion_list = ["gini", "entropy"]
max_feature_list = ["auto", "sqrt", "log2", None]

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3,56))

# Combination list
combinations_list = [list(x) for x in product(criterion_list, max_feature_list, max_depth_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)


```

```python
# Confirm how many hyperparameter combinations & print
number_combs = len(combinations_list)
print(number_combs)

# Sample and visualise specified combinations
for x in [50, 500, 1500]:
    sample_and_visualize_hyperparameters(x)
    
# Sample all the hyperparameter combinations & visualise
sample_and_visualize_hyperparameters(number_combs)



```

> The function sample_and_visualize_hyperparameters() takes a single argument (number of combinations to sample) and then randomly samples hyperparameter combinations, just like you did in the last exercise! The function will then visualize the combinations.
> If you want to see the function definition, you can use Python’s handy inspect library, like so:
> print(inspect.getsource(sample_and_visualize_hyperparameters))


![](https://i.imgur.com/5Hh5xev.png)

![](https://i.imgur.com/5wz9Hgx.png)


![](https://i.imgur.com/6iKGYma.png)


### Random Search in scikit-learn

* Compare to grid search
	* decide how many samples to take

* key difference 
	* n_iter: which is number of samples
	* param_distribution : défaut - uniform distribution

* Example
```python
# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1, 2, 150), 'min_samples_leaf': list(range(20, 65))} 

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator = GradientBoostingClassifier(),
    param_distributions = param_grid,
    n_iter = 10,
    scoring='accuracy', n_jobs=4, cv = 5, refit=True, return_train_score = True)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])


```


```python

# Create the parameter grid
param_grid = {'max_depth': list(range(5,26)), 'max_features': ['auto' , 'sqrt']} 

# Create a random search object
random_rf_class = RandomizedSearchCV(
    estimator = RandomForestClassifier(n_estimators=80),
    param_distributions = param_grid, n_iter = 5,
    scoring='roc_auc', n_jobs=4, cv = 3, refit=True, return_train_score = True)

# Fit to the training data
random_rf_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_rf_class.cv_results_['param_max_depth'])
print(random_rf_class.cv_results_['param_max_features'])


```


### Comparing Grid and Random search

* More data
* how many hyperparameters
```python
# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Print result
print(grid_combinations_chosen)

# Sample grid coordinates
grid_combinations_chosen = combinations_list[0:300]

# Create a list of sample indexes
sample_indexes = list(range(0,len(combinations_list)))

# Randomly sample 300 indexes
random_indexes = np.random.choice(sample_indexes, 300, replace=False)

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]

# Use indexes to create random sample
random_combinations_chosen = [combinations_list[index] for index in random_indexes]

# Call the function to produce the visualization
visualize_search(grid_combinations_chosen, random_combinations_chosen)

```

![](https://i.imgur.com/4ap785P.png)


## Lesson 4: Informed Search
* Uninformed search : Where each iteration of hyper parameter tuning does not learn from the previous iterations

### Coarse to Fine Tuning
	* Process
		1. Random search
		2. Find promising areas
		3. Grid search in the smaller area
		4. Continue until optimal score obtained

* why coarse to Fine
	* Utilizes the advantages of grid and random search
		* wide search to begin with
		* deeper search once you know where a good spot is likely to be
	* Better spending of time and computational efforts mean you can iterate quicker

* Example
	* max_depth : 1-65
	* min_sample_list : 3-17
	* learn_rate_list : 150 values between 0.01 and 150
```python
combinations_list = [list(x) for x in product(max_depth_list, min_sample_list, learn_rate_list)]
print(len(combination_list))
134400

```


```python

# Confirm the size of the combinations_list
print(len(combinations_list))

# Sort the results_df by accuracy and print the top 10 rows
print(results_df.sort_values(by='accuracy', ascending=False).head(10))

# Confirm which hyperparameters were used in this search
print(results_df.columns)

# Call visualize_hyperparameter() with each hyperparameter in turn
visualize_hyperparameter('max_depth')
visualize_hyperparameter('min_samples_leaf')
visualize_hyperparameter('learn_rate')


```

> import inspect
> print(inspect.getsource(visualize_hyperparameter))


![](https://i.imgur.com/xrICwJ2.png)

![](https://i.imgur.com/9J0TjHb.png)

![](https://i.imgur.com/56JAAgJ.png)


* Example : Coarse to Fine Iterations
```
# Use the provided function to visualize the first results
visualize_first() 
```


![](https://i.imgur.com/SQtKXu2.png)

![](https://i.imgur.com/qaSmxHX.png)


```python
# Use the provided function to visualize the first results
# visualize_first()

# Create some combinations lists & combine:
max_depth_list = list(range(1,21))
learn_rate_list = np.linspace(0.001,1,50)

# Call the function to visualize the second results
visualize_second()


```

### Bayesian Statistics

* Bayes Rule
> A statistical method of using new evidence to iteratively update our beliefs about some outcome

![](https://i.imgur.com/yV4X7zS.png)


* Steps
	* pick a hyper parameter combination
	* build a model
	* Get new evidence ( the score of the model)
	* update our beliefs and chose better hyper parameters next round

* “Hyperopt”
	* Set the domain : our grid
		* simple numbers
		* choose from a list
		* distribution of values
	* set the optimization algorithm
	* objective function to minimize : we will use 1-Accuracy

* Example : Bayes rule in python
```python
# Assign probabilities to variables 
p_unhappy = 0.15
p_unhappy_close = 0.35

# Probabiliy someone will close
p_close = 0.07

# Probability unhappy person will close
p_close_unhappy = (0.35 * 0.07) / 0.15
print(p_close_unhappy)


```

* Example : Bayesian Hyperparameter tuning with Hyper
```python
# Set up space dictionary with specified hyperparameters
space = {'max_depth': hp.quniform('max_depth', 2, 10, 2),'learning_rate': hp.uniform('learning_rate', 0.001, 0.9)}

# Set up objective function
def objective(params):
    params = {'max_depth': int(params['max_depth']),'learning_rate': params['learning_rate']}
    gbm_clf = GradientBoostingClassifier(n_estimators=100, **params) 
    best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
    loss = 1 - best_score
    return loss

# Run the algorithm
best = fmin(fn=objective,space=space, max_evals=20, rstate=np.random.RandomState(42), algo=tpe.suggest)
print(best)


```


### Genetic Algorithm
* How genetic evolution works
	1. There are many creatures existing (‘offspring’)
	2. The strongest creatures survive and pair off
	3. There is some ‘crossover’ as they form offspring
	4. There are random mutations to some of the offspring
		1. Therese mutations sometimes help give some offspring an advantage
	5. Go back to (1)

* “TPOT”
	* Key  arguments
		* generation : iteration
		* population_size : the number of models to keep after each iteration
		* offspring_size : number of models to produce in each iteration
		* mutation_rate : the proportion of pipelines to apply randomness to
		* crossover_rate : the proportion of pipelines to breed each iteration 
		* scoring
		* cv
	
> why no algorithm? it does all?

* Example
```python
# Assign the values outlined to the inputs
number_generations = 3
population_size = 4
offspring_size = 3
scoring_function = 'accuracy'

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size, 
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=2, cv=2)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


Warning: xgboost.XGBClassifier is not available and will not be used by TPOT.
Generation 1 - Current best internal CV score: 0.7575064376609415
Generation 2 - Current best internal CV score: 0.7750693767344183
Generation 3 - Current best internal CV score: 0.7750693767344183

Best pipeline: BernoulliNB(input_matrix, alpha=0.1, fit_prior=True)
0.76
```


```python
# Create the tpot classifier 
tpot_clf = TPOTClassifier(generations=2, population_size=4, offspring_size=3, scoring='accuracy', cv=2,
                          verbosity=2, random_state=99)

# Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Score on the test set
print(tpot_clf.score(X_test, y_test))


```


## Wrap up
* Learned
	* some hyper parameter are better to start with than others
	* there are silly values you can set for hyper parameters
	* you need to beware of conflicting hyperparameters
	* best practice is specific to algorithm and their hyperparameters
* Grid search
* Random search
* informed search
	* ‘coarse to Fine’
	* bayesian hyperparameter tuning, updating beliefs using evidence on model performance
	* Genetic algorithm, evolving your models over generations

