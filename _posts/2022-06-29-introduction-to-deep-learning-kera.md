---
layout: post
title: Introduction to Deep Learning with Keras
date: 2022-06-29
tags: datacamp, python, keras, deep learning
categories: datacamp python keras deep learning
comments: true
---

## *Introducing Keras*
* High level deep learning framework
![](https://i.imgur.com/9rl6KzU.png)

* Runs on top of other frameworks
* Less code
* use `Tensorflow` for low level features

* feature Engineering
![](https://i.imgur.com/o2bBR4h.png)

* Suitable for unstructured data
	* not easy to put into table for explainable computation
* neural networks
	* images
	* don’t care about why the network know it’s a cat or dog
	* benefit from CNN

### Neural network
![](https://i.imgur.com/6vKa3yw.png)

![](https://i.imgur.com/up6QlNk.png)


* Gradient descent
![](https://i.imgur.com/veXKk6g.png)

	* back-propagation
* Sequential API
	* building the model as a stack of layers
	![](https://i.imgur.com/gEBPxmT.png)

	* sample
	![](https://i.imgur.com/frMD1ud.png)

	![](https://i.imgur.com/kvjirwL.png)

	* Visualize the parameters
	![](https://i.imgur.com/ArSrWEP.png)

	* Example
![](https://i.imgur.com/NQw2GfO.png)

```python
# Instantiate a new Sequential model
model = Sequential()

# Add a Dense layer with five neurons and three inputs
model.add(Dense(5, input_shape=(3,), activation="relu"))

# Add a final Dense layer with one neuron and no activation
model.add(Dense(1))

# Summarize your model
model.summary()
```


	* Example
> Your training data consist of measurements taken at time steps from *-10 minutes before the impact region to +10 minutes after*. Each time step can be viewed as an X coordinate in our graph, which has an associated position Y for the meteor orbit at that time step.
	![](https://i.imgur.com/iZ40hz9.jpg)


```python
# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))

# End your model with a Dense layer and no activation
model.add(Dense(1))



# Compile your model
model.compile(optimizer = 'adam', loss = 'mse')

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps, y_positions, epochs = 30)

# Evaluate your model 
print("Final loss value:",model.evaluate(time_steps, y_positions))


# Predict the twenty minutes orbit
twenty_min_orbit = model.predict(np.arange(-10, 11))

# Plot the twenty minute orbit 
plot_orbit(twenty_min_orbit)


```

## Going Deeper
### Binary Classification
![](https://i.imgur.com/Dj4epfm.png)


![](https://i.imgur.com/gpn9oBu.png)


* sigmoid function
![](https://i.imgur.com/j0jx7qM.png)

* 
![](https://i.imgur.com/J6dWzqw.png)


![](https://i.imgur.com/CqpXWkR.png)


![](https://i.imgur.com/4vHaEtG.png)



* Example : 
> Your goal is to distinguish between real and fake dollar bills. In order to do this, the dataset comes with 4 features: variance,skewness,kurtosis and entropy. These features are calculated by applying mathematical operations over the dollar bill images. The labels are found in the dataframe’s class column.
>  ![](https://i.imgur.com/IPrX2rx.png)


```python
# Import seaborn
import seaborn as sns

# Use pairplot and set the hue to be our class column
sns.pairplot(banknotes, hue='class') 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations per class
print('Observations per class: \n', banknotes['class'].value_counts())


```

![](https://i.imgur.com/kEPdhXi.png)


```python
# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Display a summary of your model
model.summary()


# Train your model for 20 epochs
model.fit(X_train, y_train, epochs = 20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test, y_test)[1]

# Print accuracy
print('Accuracy:', accuracy)

```



### Multi-class classification
* Throwing darts
	* datasets
	![](https://i.imgur.com/18lc8nQ.png)

	![](https://i.imgur.com/7OCXDT6.png)

	* architecture
![](https://i.imgur.com/ek14fii.png)
![](https://i.imgur.com/Z7iuYLD.png)

	* softmax : 總和為1

```python
# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation='softmax'))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Print the label encoded competitors
print('Label encoded competitors: \n',darts.competitor.head())

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Import to_categorical from keras utils module
from keras.utils import to_categorical

coordinates = darts.drop(['competitor'], axis=1)
# Use to_categorical on your labels
competitors = to_categorical(darts.competitor)

# Now print the one-hot encoded labels
print('One-hot encoded competitors: \n',competitors)


# Fit your model to the training data for 200 epochs
model.fit(coord_train, competitors_train, epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

```


```python
# Predict on coords_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the position of highest probability from each pred vector
preds_chosen = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds_chosen):
  print("{:25} | {}".format(pred,competitors_small_test[i]))

Raw Model Predictions                         | True labels
[0.34438723 0.00842557 0.63167274 0.01551455] | [0. 0. 1. 0.]
[0.0989717  0.00530467 0.07537904 0.8203446 ] | [0. 0. 0. 1.]
[0.33512568 0.00785374 0.28132284 0.37569773] | [0. 0. 0. 1.]
[0.8547263  0.01328656 0.11279515 0.01919206] | [1. 0. 0. 0.]
[0.3540977  0.00867271 0.6223853  0.01484426] | [0. 0. 1. 0.]
Rounded Model Predictions | True labels
                        2 | [0. 0. 1. 0.]
                        3 | [0. 0. 0. 1.]
                        3 | [0. 0. 0. 1.]
                        0 | [1. 0. 0. 0.]
                        2 | [0. 0. 1. 0.]
```

* Multi-label classification
	* Example 1
	![](https://i.imgur.com/00t3gk7.png)

		* architecture
			* sigmoid : 每一輸出介於0~1
* 
![](https://i.imgur.com/fHT4vy5.png)

	* several classifier : one-vs-rest 

* Example :
> You’re going to automate the watering of farm parcels by making an intelligent irrigation machine. Multi-label classification problems differ from multi-class problems in that each observation can be labeled with zero or more classes. So classes/labels are not mutually exclusive, you could water all, none or any combination of farm parcels based on the inputs.
> To account for this behavior what we do is have an output layer with as many neurons as classes but this time, unlike in multi-class problems, each output neuron has a sigmoid activation function. This makes each neuron in the output layer able to output a number between 0 and 1 independently.
> ![](https://i.imgur.com/uZag2Yt.jpg)


```python
# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64, input_shape=(20,), activation='relu'))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3, activation='sigmoid'))

# Compile your model with binary crossentropy loss
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.summary()


```


> An output of your multi-label model could look like this: [0.76 , 0.99 , 0.66 ]. If we round up probabilities higher than 0.5, this observation will be classified as containing all 3 possible labels [1,1,1]. For this particular problem, this would mean watering all 3 parcels in your farm is the right thing to do, according to the network, given the input sensor measurements.


```python
# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs=100, validation_split=0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test, parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)



```


### Keras callbacks
* Record the results after each epoch 
![](https://i.imgur.com/NfndVSL.png)

	* for validation data

![](https://i.imgur.com/o6wfmUg.png)

![](https://i.imgur.com/cmQZ2sJ.png)


* Early stopping
	* 不在改善時停止
![](https://i.imgur.com/AZ0fsic.png)


* Checkpoint
	* 保存訓練過程中的模型

![](https://i.imgur.com/3X36pO3.png)


* Example
```python
# Train your model and save its history
h_callback = model.fit(X_train, y_train, epochs = 50,
                    validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])

# Plot train vs test accuracy during training
plot_accuracy(h_callback.history['acc'], h_callback.history['val_acc'])

# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor='val_acc', 
                                patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
          epochs=1000, validation_data=(X_test, y_test),
          callbacks=[monitor_val_acc])

```


```python

# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor = 'val_acc', patience = 3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint('best_banknote_model.hdf5', save_best_only = True)

# Fit your model for a stupid amount of epochs
h_callback = model.fit(X_train, y_train,
                    epochs = 1000000000000,
                    callbacks = [monitor_val_acc, modelCheckpoint],
                    validation_data = (X_test, y_test))


```
* Word2Vec 是否為deep learning model?

![](https://i.imgur.com/Jd7Faxt.png)


## *Improving Your Model Performance*
### Learning curve
* loss learning curve
* accuracy learning curve

![](https://i.imgur.com/ikRF0QS.png)


* sample
``` python
# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape = (64,), activation = 'relu'))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10, activation = 'softmax'))

# Compile your model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Test if your model is well assembled by predicting before training
print(model.predict(X_train))


```


```python
def plot_loss(loss,val_loss):
  plt.figure()
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper right')
  plt.show()

# Train your model for 60 epochs, using X_test and y_test as validation data
h_callback = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the h_callback object loss and val_loss to plot the learning curve
plot_loss(h_callback.history['loss'], h_callback.history['val_loss'])



```


![](https://i.imgur.com/tXyvdmH.png)



```python
for size in training_sizes:
    # Get a fraction of training data (we only care about the training data)
    X_train_frac, y_train_frac = X_train[:size], y_train[:size]

    # Reset the model to the initial weights and train it on the new training data fraction
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs = 50, callbacks = [early_stop])

    # Evaluate and store both: the training data fraction and the complete test set results
    train_accs.append(model.evaluate(X_train_frac, y_train_frac)[1])
    test_accs.append(model.evaluate(X_test, y_test)[1])
    
# Plot train vs test accuracies
plot_results(train_accs, test_accs)


```

### Activation functions

![](https://i.imgur.com/osZcLkU.png)

1. Sigmoid : 0~1
![](https://i.imgur.com/5PEJyle.png)


2. Tanh : -1~1

![](https://i.imgur.com/fllPNNq.png)

3. ReLu

![](https://i.imgur.com/3w4Du3a.png)

4. Leaky Relu

![](https://i.imgur.com/wPleqzF.png)


* Binary Classification
![](https://i.imgur.com/rm3XM8h.png)


![](https://i.imgur.com/1TCEVH5.png)


* Which activation function to use?
	* No magic function
	* different properties
	* depends on our problem
	* Goal to achieve in a given layer
	* ReLU are a good first choice
		* train fast
		* good for most problems
	* Sigmoid not recommended for deep models
	* Tune with experimentation

* Comparing activation functions
![](https://i.imgur.com/uDQ2KBJ.png)

![](https://i.imgur.com/8laxM7n.png)

![](https://i.imgur.com/Yru6nwn.png)


* sample 
```python
# Activation functions to try
activations = ['relu', 'leaky_relu', 'sigmoid', 'tanh']

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model and store the history results
  h_callback = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20, verbose=0)
  activation_results[act] = h_callback


```

> What you coded in the previous exercise has been executed to obtain theactivation_results variable, this time *100 epochs were used instead of 20*. This way you will have more epochs to further compare how the training evolves per activation function.
> For every h_callback of each activation function in activation_results:
> * The h_callback.history[‘val_loss’] has been extracted.
> * The h_callback.history[‘val_acc’] has been extracted.
> Both are saved into two dictionaries: val_loss_per_function and val_acc_per_function.

```python

# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()


```

![](https://i.imgur.com/DAeiVRG.png)



### *Batch size and batch normalization*
* Batch and mini-batches
![](https://i.imgur.com/ncCrhMK.png)


![](https://i.imgur.com/sRvZ2g0.png)


* If data is 9, batch size = 3, 3 mini-batches
* 1 epoch update 3 times weight due to 3 mini-batches

* Mini-batches 
	* pros
		* training fast
		* make batch stored in memory
		* noise can help networks reach a lower error, escaping local optimization value
	* cons
		* more iterations
		* need to find out a good batch size
![](https://i.imgur.com/drq3aGS.png)

		* stochastic 隨機梯度下降
			* batch size = 1
		* batch size smaller, iterations more

* Batch Normalization
	* (data - mean) / standard deviation
![](https://i.imgur.com/fUfh91J.png)

![](https://i.imgur.com/lGYUATe.png)

	* improves gradient flow 
	* allow higher learning rate
	* reduces dependence on weight initializations
	* acts as an unintended from of regularization
	* limits internal covariate shift
		* 某一層在學習時 會依賴於前一層的輸出

![](https://i.imgur.com/T99dfjY.png)


* sample 
``` python
# Import batch normalization from keras layers
from keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

```

```python

# Train your standard model, storing its history callback
h1_callback = standard_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history callback
h2_callback = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Call compare_histories_acc passing in both model histories
compare_histories_acc(h1_callback, h2_callback)


```

![](https://i.imgur.com/y4vayYN.png)



### *Hyperparameter tuning*
* Hyperparameter
	* number of layers
	* number of neurons per layer
	* layer order
	* layer activations
	* batch sizes
	* learning rates
	* optimizer
	* …

* use Sklearn RamdomizedSearchCV
* Turn a Keras model into a Sklearn estimator
![](https://i.imgur.com/OZkfPiP.png)


* cross-validation
![](https://i.imgur.com/tvQGNXP.png)




* Tips for neural networks hyper parameter tuning
	* Random search is preferred over grid search
	* don’t use many epochs
	* use a smaller sample of your dataset
	* play with batch sizes, activations, optimizers and learning rates

![](https://i.imgur.com/FY4ezIt.png)


* 調整神經元數量與層數
![](https://i.imgur.com/Z4l7K1N.png)


![](https://i.imgur.com/kZFAbpT.png)


> You’ve seen that the first step to turn a model into a sklearn estimator is to build a function that creates it. The definition of this function is important since hyperparameter tuning is carried out by varying the arguments your function receives.

```python
# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    
    # Create your binary classification model  
    model = Sequential()
    model.add(Dense(128, input_shape = (30,), activation = activation))
    model.add(Dense(256, activation = activation))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


```

> Since fitting the RandomizedSearchCV object would take too long, the results you’d get are printed in the show_results() function. You could try random_search.fit(X,y) in the console yourself to check it does work after you have built everything else, but you will probably timeout the exercise (so copy your code first if you try this or you can lose your progress!).

```python


# Creates a model given an activation and learning rate
def create_model(learning_rate, activation):
  
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr = learning_rate)
    
    # Create your binary classification model  
    model = Sequential()
    model.add(Dense(128, input_shape = (30,), activation = activation))
    model.add(Dense(256, activation = activation))
    model.add(Dense(1, activation = ‘sigmoid’))
    
    # Compile your model with your optimizer, loss, and metrics
    model.compile(optimizer = opt, loss = ‘binary_crossentropy’, metrics = [‘accuracy’])
    return model



Best: 
    0.975395 using {learning_rate: 0.001, epochs: 50, batch_size: 128, activation: relu} 
    Other: 
    0.956063 (0.013236) with: {learning_rate: 0.1, epochs: 200, batch_size: 32, activation: tanh} 
    0.970123 (0.019838) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: tanh} 
    0.971880 (0.006524) with: {learning_rate: 0.01, epochs: 100, batch_size: 128, activation: tanh} 
    0.724077 (0.072993) with: {learning_rate: 0.1, epochs: 50, batch_size: 32, activation: relu} 
    0.588752 (0.281793) with: {learning_rate: 0.1, epochs: 100, batch_size: 256, activation: relu} 
    0.966608 (0.004892) with: {learning_rate: 0.001, epochs: 100, batch_size: 128, activation: tanh} 
    0.952548 (0.019734) with: {learning_rate: 0.1, epochs: 50, batch_size: 256, activation: relu} 
    0.971880 (0.006524) with: {learning_rate: 0.001, epochs: 200, batch_size: 128, activation: relu}
    0.968366 (0.004239) with: {learning_rate: 0.01, epochs: 100, batch_size: 32, activation: relu}
    0.910369 (0.055824) with: {learning_rate: 0.1, epochs: 100, batch_size: 128, activation: relu}

```


> Time to train your model with the best parameters found: *0.001* for the *learning rate*, *50 epochs*, *a 128 batch_size* and *relu activations*.
> The create_model() function from the previous exercise is ready for you to use. X and y are loaded as features and labels.

```python
# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn = create_model(learning_rate = 0.001, activation = 'relu'), epochs = 50, 
                        batch_size = 128, verbose = 0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv = 3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())


The mean accuracy was: 0.9718834066666666
With a standard deviation of: 0.002448915612216046

```


## *Advanced Model Architectures*
### *Tensors, layers, and autoencoders*
* Accessing Keras layers
![](https://i.imgur.com/0SuU8YE.png)

* What tensors
![](https://i.imgur.com/ervSN0C.png)



![](https://i.imgur.com/XK9DSM1.png)


* New architecture
	* auto encoder
	* 
![](https://i.imgur.com/AxG3c6K.png)


		* input = output
		* lower dimensional representation
		* reduce neurons
		* use case
			* dimensionality reduction
			* de-noising data:
				* if trains with clean data, irrelevant noise will be filtered out during reconstruction
			* anomaly detection
				* 輸入異常值時, 會無法準確輸出
* sample

![](https://i.imgur.com/hBNxKnf.png)


![](https://i.imgur.com/aXiRoj3.png)


* sample
> If you have already built a model, you can use the model.layers and the keras.backend to build functions that, provided with a valid input tensor, return the corresponding output tensor.
> This is a useful tool when we want to obtain the output of a network at an intermediate layer.
> For instance, if you get the input and output from the first layer of a network, you can build an inp_to_out function that returns the result of carrying out forward propagation through only the first layer for a given input tensor.

```python

# Import keras backend
import keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp], [out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))


```

* *Neural separation*
```python
for i in range(0, 21):
    # Train model for 1 epoch
    h = model.fit(X_train, y_train, batch_size = 16, epochs = 1, verbose = 0)
    if i%4==0: 
      # Get the output of the first layer
      layer_output = inp_to_out([X_test])[0]
      
      # Evaluate model accuracy for this epoch
      test_accuracy = model.evaluate(X_test, y_test)[1] 
      
      # Plot 1st vs 2nd neuron output
      plot()

```

![](https://i.imgur.com/7uxyhQq.png)


* auto encoder
```python
# Start with a sequential model
autoencoder = Sequential()

# Add a dense layer with input the original image pixels and neurons the encoded representation
autoencoder.add(Dense(32, input_shape=(784, ), activation="relu"))

# Add an output layer with as many neurons as the orginal image pixels
autoencoder.add(Dense(784, activation = "sigmoid"))

# Compile your model with adadelta
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

# Summarize your model structure
autoencoder.summary()


```

* de-noise like an auto encoder
```python
# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)


# Build your encoder by using the first layer of your autoencoder
encoder = Sequential()
encoder.add(autoencoder.layers[0])

# Encode the noisy images and show the encodings for your favorite number [0-9]
encodings = encoder.predict(X_test_noise)
show_encodings(encodings, number = 1)

# Predict on the noisy images with your autoencoder
decoded_imgs = autoencoder.predict(X_test_noise)

# Plot noisy vs decoded images
compare_plot(X_test_noise, decoded_imgs)


```


![](https://i.imgur.com/w9lNsmz.png)



![](https://i.imgur.com/wf909wW.png)


### Intro to CNNs
* filter/kernel
* 3 x 3 kernel
![](https://i.imgur.com/Seh7V6c.gif)

![](https://i.imgur.com/jKnhKBQ.gif)


![](https://i.imgur.com/eDItV92.png)

* input_shape : width, height, channels

![](https://i.imgur.com/mT0xGEq.png)


* ResNet50
	* preprocessing images for ResNet50

![](https://i.imgur.com/8jw7GZZ.png)

	* use the rest50 model in keras

![](https://i.imgur.com/6zoWWPh.png)



* sample
```python

# Import the Conv2D and Flatten layers and instantiate model
from keras.layers import Conv2D,Flatten
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3 
model.add(Conv2D(32, kernel_size = 3, input_shape = (28, 28, 1), activation = 'relu'))

# Add a convolutional layer of 16 filters of size 3x3
model.add(Conv2D(16, kernel_size = 3, activation = 'relu'))

# Flatten the previous layer output 
model.add(Flatten())

# Add as many outputs as classes with softmax activation
model.add(Dense(10, activation = 'softmax'))


# Obtain a reference to the outputs of the first layer
first_layer_output = model.layers[0].output

# Build a model using the model input and the first layer output
first_layer_model = Model(inputs = model.layers[0].input, outputs = first_layer_output)

# Use this model to predict on X_test
activations = first_layer_model.predict(X_test)

# Plot the first digit of X_test for the 15th filter
axs[0].matshow(activations[0,:,:,14], cmap = 'viridis')

# Do the same but for the 18th filter now
axs[1].matshow(activations[0,:,:,17], cmap = 'viridis')
plt.show()


````


![](https://i.imgur.com/VryEFFq.png)


* ResNet50
[image:D8FD5A76-FD84-4A6A-9B69-F3975710EFEE-584-00001CD2AF18B9AB/dog.png]
```python

# Import image and preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

# Load the image with the right target size for your model
img = image.load_img(img_path, target_size = (224, 224))

# Turn it into an array
img_array = image.img_to_array(img)

# Expand the dimensions of the image, this is so that it fits the expected model input format
img_expanded = np.expand_dims(img_array, axis = 0)

# Pre-process the img in the same way original images were
img_ready = preprocess_input(img_expanded)


```

![](https://i.imgur.com/2pgHs66.png)

```python
# Instantiate a ResNet50 model with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Predict with ResNet50 on your already processed img
preds = model.predict(img_ready)

# Decode the first 3 predictions
print('Predicted:', decode_predictions(preds, top=3)[0])

Predicted: [('n02088364', 'beagle', 0.8280003), ('n02089867', 'Walker_hound', 0.12915272), ('n02089973', 'English_foxhound', 0.03711732)]


```


### LSTM
![](https://i.imgur.com/uEP3WGc.png)

![](https://i.imgur.com/W4Fq8Jz.png)

* image captioning
![](https://i.imgur.com/vLt4Qr4.png)


* need to covert string into number -> embedding 
![](https://i.imgur.com/NR4RMWb.png)

* 
![](https://i.imgur.com/vJZW6uw.png)

* to numbers
![](https://i.imgur.com/4xW8Wxw.png)

![](https://i.imgur.com/lkGA4IV.png)


* Example :*Text prediction with LSTMs*
```python

# Split text into an array of words
words = text.split()

# Make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
  sentences.append(' '.join(words[i-4:i]))
  
# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))


# Split text into an array of words
words = text.split()

# Make sentences of 4 words each, moving one word at a time
sentences = []
for i in range(4, len(words)):
  sentences.append(' '.join(words[i-4:i]))
  
# Instantiate a Tokenizer, then fit it on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Turn sentences into a sequence of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5],sequences[:5]))


```


```python

def predict_text(test_text, model = model):
  if len(test_text.split()) != 3:
    print('Text input should be 3 words!')
    return False
  
  # Turn the test_text into a sequence of numbers
  test_seq = tokenizer.texts_to_sequences([test_text])
  test_seq = np.array(test_seq)
  
  # Use the model passed as a parameter to predict the next word
  pred = model.predict(test_seq).argmax(axis = 1)[0]
  
  # Return the word that maps to the prediction
  return tokenizer.index_word[pred]


In [1]:
predict_text('meet revenge with')
Out[1]:
'revenge'
In [2]:
predict_text('the course of')
Out[2]:
'history'
In [3]:
predict_text('strength of the')
Out[3]:
'spirit'

```
