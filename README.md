# TCDML_income

General Overview of the process, what I do, and why:

- Import all functions used, so that I can use them
- Define function for pre-processing data, so that the data we load in can get cleaned
- Define columns used in the dataset, to aid creation of the dataframe
- Load in training & testing data, to be cleaned
- Apply column names to the data to index
- Pre-process training and testing data in the preprocessing function, creating a new "Low Population" column in place of the "Size of City" column, and replacing each individual income value with its natural log. (Also, the training data has all rows with no data in the Age and/or Year of Record columns deleted.) This cleans and improves the quality of our datasets through normalisation and removal of outliers.
- Run pd.get_dummies on the data to expand non-numerical fields into a one-hot encoded field. This further improves the quality of the dataset and prepares it for training/testing.
- Fill any remaining not-a-number datapoints in either dataset with the mean value of that column. This minimizes the impact of not-a-number values, which are outliers.
- Reduce both datasets to only include the columns that are not unique to that dataset. This ensures that neither dataset contains a column that is not in the other dataset. At this stage, the only missing data in the testing dataset should be the income.
- Pull the income column from the training data an store it in another variable, then delete it from the other dataset. This is used to train the Linear Regression model.
- Initialise, then train the Linear Regression model. This is done to predict the new data.
- Use the trained Linear Regression model to predict the income for the test data.
- Use np.exp() to reverse the log function, in order to find the prediced income value.
- Read in the .csv file for submission
- Put the expected income values into their proper index in the .csv's dataframe
- Save the dataframe, overwriting the .csv file with the updated results.
