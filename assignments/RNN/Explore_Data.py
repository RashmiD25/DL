
## Set Up the Environment
import os
import numpy as np
import pandas as pd              
import matplotlib.pyplot as plt 
import random

# Configure the notebook to display plots
%matplotlib inline

# Define the path to the data folder
data_dir = '/dli/task/data/hx_series'
# training data inputs: x and targets: y
x_train_path = os.path.join(data_dir, 'X_train.hdf')
y_train_path = os.path.join(data_dir, 'y_train.hdf')

# validation data inputs: x and targest: y
x_valid_path = os.path.join(data_dir, 'X_test.hdf')
y_valid_path = os.path.join(data_dir, 'y_test.hdf')

## Load the Data with <i>pandas</i>
X_train = pd.read_hdf(x_train_path)
y_train = pd.read_hdf(y_train_path)
X_valid = pd.read_hdf(x_valid_path)
y_valid = pd.read_hdf(y_valid_path)
print('data load complete')

## Visualize the Data

X_train

y_train

### 1.3.1 Example: View Data from a Single Encounter

X_train.index.levels[0]

# Select a random patient encounterID from a list of all the encounterID values
eIdx = random.choice(list(X_train.index.levels[0]))

# Specify a few variables to look at
variables = [
    'Age','Heart rate (bpm)','PulseOximetry','Weight',
    'SystolicBP','DiastolicBP','Respiratory rate (bpm)',
    'MotorResponse','Capillary refill rate (sec)'
]

# Note that the full list of variables can be constructed using
# list(X_train.columns.values)

# Have a look at the variables for the selected patient
print('encounterID = {}'.format(eIdx))
print('number of observations = {}'.format(X_train.loc[eIdx].index.shape[0]))
print('max absoluteTime value = {} hours'.format(X_train.loc[eIdx].index[-1]))
X_train.loc[eIdx, variables]

_train.loc[eIdx, "Heart rate (bpm)"].plot()
plt.ylabel("Heart rate (bpm)")
plt.xlabel("Hours since first encounter")
plt.show()

#Exercise: Observation Count Histogram

eIdx_list = X_train.index.levels[0] #FIXME
print('eIdx_list created of length {} (Sanity check: this length should be 5000)'.format(len(eIdx_list)))

nobs_list = [X_train.loc[ix].index.shape[0] for ix in eIdx_list]
print('list of observation counts created of length {} (Sanity check: this length should also be 5000)'.format(len(nobs_list)))

plt.hist(nobs_list,range=(0,1000))
plt.title("Observations per Patient Encounter")
plt.show()

print('The mean = {}'.format(np.mean(nobs_list)))
print('The median = {}'.format(np.median(nobs_list)))
<a name='01_ex_time'></a>


eIdx_list = X_train.index.levels[0] #FIXME
print('eIdx_list created of length {} (Sanity check: this length should be 5000)'.format(len(eIdx_list)))

timespan_list = [X_train.loc[ix].index[-1] for ix in eIdx_list] #FIXME
print('timespan_list created of length {} (Sanity check: this length should be 5000)'.format(len(timespan_list)))

plt.hist(timespan_list,range=(0,500))
plt.title("Hours per Patient Encounter")
plt.show()

print('The mean = {}'.format(np.mean(timespan_list)))
print('The median = {}'.format(np.median(timespan_list)))

# Save a *pandas* DataFrames
Before moving to the next notebook, save the *pandas* DataFrames to reload into the next notebook.
# Save the DataFrame's for use in other notebooks
X_train.to_pickle('X_train.pkl')
y_train.to_pickle('y_train.pkl')
X_valid.to_pickle('X_valid.pkl')
y_valid.to_pickle('y_valid.pkl')

