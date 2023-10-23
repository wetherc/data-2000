# Linear Models

This homework is due on or before Tuesday 24 October, 11:59pm Eastern time. Publish your code to GitHub and provide a link to it in your Canvas submission.

For this problem set, we will use the Craigslist used vehicle dataset from our 03 Oct lab. As a reminder, you can load it into your development environment with:

```python
import pandas as pd

car_data_raw = pd.read_csv("https://cdn.c18l.org/vehicles_lab.csv")
```

## Part 1: Feature Selection

Our dataset contains the following features:
  - region
  - price
  - year
  - manufacturer
  - model
  - condition
  - cylinders
  - fuel
  - odometer
  - title_status
  - transmission
  - drive
  - size
  - type
  - paint_color
  - description
  - state
  - lat
  - long
  - posting_date

`price` is the column we will use as our label; the remaining columns are all possible inputs to our model.

**Create a dataframe with `price` and between 5 and 7 additional columns from our original dataset that you want to use as features for a predictive model. Explain your choices.**

## Part 2: Data Cleaning

Based on the dataset that you created for Part 1, **normalize any numeric features, dummy- or one-hot encode any categorical features, and remove any outliers or spurious records. Explain your choices.** 

## Part 3: Feature Engineering

Based on the dataset that you created for Part 2, **create two or more new engineered feature columns and explain why you chose to create these.**

## Part 4: Multinomial classification

Based on the dataset that you created for Part 3:

  - Create a new `price_bin` column that bins vehicle price into $10,000 bands. E.g., `$0-$9,999`, `$10,000-19,999`, and so forth. Use [the `cut()` function](https://pandas.pydata.org/docs/reference/api/pandas.cut.html) in Pandas to do this;
  - Split your dataset into training and testing samples at an 80:20 ratio;
  - Train a multinomial logistic regression model to predict which price band each car will fall into. Use [scikit-learn's LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to do this. For a multinomial classification problem, use the `multi_class='multinomial'` parameter when fitting your model.

**What is your model's accuracy on both the training and testing datasets?** You can use [scikit-learn's accuracy_score() function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) to determine this.
