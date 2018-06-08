# Backorder Prediction Modeling
Personal Data Analysis project of Backorder Prediction using Kaggle dataset 


## Data
I used [data](https://www.kaggle.com/tiredgeek/predict-bo-trial) from Kaggle, which had 1.9 million observations of parts in an 8 week period. The source of the data is unreferenced.

-   **Outcome**: whether the part went on backorder
-   **Predictors**: Current inventory, sales history, forecasted sales, recommended stocking amount, part risk flags etc. (22 predictors in total)


## Key considerations of the data:

-   **Imbalanced outcome**: Only 0.7% of parts actually go on backorder.
-   **Outliers and skewed predictors**: Part quantities (stock, sales etc.) can be on very different scales.
-   **Missing data**: A few variables have data that are missing (not at random).
-   **n&gt;&gt;p**: There are many observations (1.9 million) relative to the number of predictors (22).


## Tasks

### Data Wrangling 
- [x] Handling inconsistent column names and datatype
- [x] Missing Data handling
- [] Removal of duplicate rows
- [] Handling columns with repetitive values
- [] Handling the outliers
- [] Write the clean data into a new file for further steps

### Data trend analysis

- [] Relationship among features
- [] Data questioning

### Model training and validation

- [] train-test split of the data
- [] Model training
- [] Model Tuning and cross validation

## Tests

## Installation
Dependencies:
``` pip install requirements.txt```


## API Reference




