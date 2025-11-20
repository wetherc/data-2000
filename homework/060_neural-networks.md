# Neural Networks

This homework is due on or before Tuesday 30 October, 11:59pm Eastern time. Publish your code to GitHub and provide a link to it in your Canvas submission.

For this problem set, we will use the CDC Diabetes Health Indicators dataset from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). You can load it into your development environment as a Pandas dataframe with:

```bash
pip install ucimlrepo
```

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# metadata
print(cdc_diabetes_health_indicators.metadata)

# variable information
print(cdc_diabetes_health_indicators.variables)
```

This dataset was created by the Centers for Disease Control and Prevention to better understand the relationship between lifestyle and diabetes in the US. Each row represents a person participating in this study.

## Part 1: Feature Selection

Our dataset contains a participant ID column, `Diabetes_binary` (which is the column we will use as our label), and 21 additional columns that can all serve as possible inputs to our model. A complete data dictionary is available at the [UCI dataset page](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).

**Create a dataframe with `Diabetes_binary` and as many additional columns from our original dataset as you feel are necessary as features for a predictive model. Explain your choices.**

## Part 2: Data Cleaning

Based on the dataset that you created for Part 1, **normalize any numeric features, dummy- or one-hot encode any categorical features, and remove any outliers or spurious records. Explain your choices.**

You can use Tensorflow's [`CategoryEncoding` preprocessing layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding) for any boolean ot categorical columns. For more on using preprocessing layers, check out this [Tensorflow tutorial](https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers#apply_the_keras_preprocessing_layers).

## Part 3: Feature Engineering

Based on the dataset that you created for Part 2, **create one or more engineered feature columns and explain why you chose to create these, or explain why you don't feel any are needed.**

## Part 4: Binary classification

Based on the dataset that you created for Part 3:

  - Split your dataset into training and testing samples at an 80:20 ratio;
  - Train a feed-forward neural network to predict whether an individual either has diabetes, or is at risk of developing diabetes.
    - This should include at least 2 hidden layers
    - The output layer should be a single neuron with a sigmoid activation function
    - The model should use a [binary crossentropy loss function](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy)
    - You can use the Adam optimizer, or another if you prefer
    - You [model's metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics) should include accuracy and F1 score

**What is your model's accuracy on both the training and testing datasets?**

  - How does the accuracy compare against the F1 score?
  - Looking at a confusion matrix of your model's predictions (i.e., the true and false positive and negative predictions), would you consider your model to be a good classifier or not? Why?