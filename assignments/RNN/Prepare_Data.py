
The initial data, in its raw form, includes variabilities that will become problematic when we process it through a model to find patterns.  The variabilities in the data that we want to mitigate are:

* Measurement values have widely varying ranges that require normalization
* Not every measurement is taken during every observation, resulting in data gaps
* The number of observations per encounter varies widely

In this notebook, we will introduce techniques to address these concerns and prepare the data for training. 
2.1 [**Normalize the Data**](#02_normalize)<br>
2.2 [**Fill Data Gaps**](#02_gaps)<br>
2.3 [**Pad Variable Length Sequences**](#02_pad)<br>
&nbsp; &nbsp; &nbsp;2.3.1 [Exercise: Padded Variable over all Patient Encounters](#02_ex_pad)<br>
2.4 [**Save a *NumPy* Array**](#02_save)<br>

import os
import numpy as np          
import pandas as pd              
import matplotlib.pyplot as plt  
import random
import tensorflow.keras as keras

# configure notebook to display plots
%matplotlib inline

# set up user paths
data_dir = '/dli/task/data/hx_series'
csv_dir = '/dli/task/csv'
# Fetch the DataFrame's loaded in the problem setup
X_train=pd.read_pickle('X_train.pkl')
y_train=pd.read_pickle('y_train.pkl')
X_valid=pd.read_pickle('X_valid.pkl')
y_valid=pd.read_pickle('y_valid.pkl')

## 2.1 Normalize the Data

# Before normalization
X_train.loc[8,['Age','Heart rate (bpm)','PulseOximetry','Weight',
    'SystolicBP','DiastolicBP','Respiratory rate (bpm)',
    'MotorResponse','Capillary refill rate (sec)']]
# create file path for csv file with metadata about variables
metadata = os.path.join(csv_dir, 'ehr_features.csv')

# read in variables from csv file (using pandas) since each varable there is tagged with a category
variables = pd.read_csv(metadata, index_col=0)

# next, select only variables of a particular category for normalization
normvars = variables[variables['type'].isin(['Interventions', 'Labs', 'Vitals'])]

# finally, iterate over each variable in both training and validation data
for vId, dat in normvars.iterrows():
    
    X_train[vId] = X_train[vId] - dat['mean']
    X_valid[vId] = X_valid[vId] - dat['mean']
    X_train[vId] = X_train[vId] / (dat['std'] + 1e-12)
    X_valid[vId] = X_valid[vId] / (dat['std'] + 1e-12)
    
# After normalization
X_train.loc[8,['Age','Heart rate (bpm)','PulseOximetry','Weight',
    'SystolicBP','DiastolicBP','Respiratory rate (bpm)',
    'MotorResponse','Capillary refill rate (sec)']]

  
## 2.2 Fill Data Gaps

# Before filling gaps
X_train.loc[8, "Heart rate (bpm)"].plot()
plt.title("Normalized Not Filled")
plt.ylabel("Heart rate (bpm)")
plt.xlabel("Hours since first encounter")
plt.show()

# first select variables which will be filled in
fillvars = variables[variables['type'].isin(['Vitals', 'Labs'])].index

# next forward fill any missing values with more recently observed value
X_train[fillvars] = X_train.groupby(level=0)[fillvars].ffill()
X_valid[fillvars] = X_valid.groupby(level=0)[fillvars].ffill()

# finally, fill in any still missing values with 0 (i.e. values that could not be filled forward)
X_train.fillna(value=0, inplace=True)
X_valid.fillna(value=0, inplace=True)

  # After filling gaps
X_train.loc[8, "Heart rate (bpm)"].plot()
plt.title("Normalized and Filled")
plt.ylabel("Heart rate (bpm)")
plt.xlabel("Hours since first encounter")
plt.show()

X_train

type(X_train)

## 2.3 Pad Variable Length Sequences

# max number of sequence length
maxlen = 500

# get a list of unique patient encounter IDs
teId = X_train.index.levels[0]
veId = X_valid.index.levels[0]

# pad every patient sequence with 0s to be the same length, 
# then transforms the list of sequences to one numpy array
# this is for efficient minibatching and GPU computations 
X_train = [X_train.loc[patient].values for patient in teId]
y_train = [y_train.loc[patient].values for patient in teId]

X_train = sequence.pad_sequences(X_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
y_train = sequence.pad_sequences(y_train, dtype='float32', maxlen=maxlen, padding='post', truncating='post')

# repeat for the validation data

X_valid = [X_valid.loc[patient].values for patient in veId]
y_valid = [y_valid.loc[patient].values for patient in veId]

X_valid = sequence.pad_sequences(X_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')
y_valid = sequence.pad_sequences(y_valid, dtype='float32', maxlen=maxlen, padding='post', truncating='post')

# print the shape of the array which will be used by the network
# the shape is of the form (# of encounters, length of sequence, # of features)
print("X_train shape: %s | y_train shape: %s" % (str(X_train.shape), str(y_train.shape)))
print("X_valid shape: %s | y_valid shape: %s" % (str(X_valid.shape), str(y_valid.shape)))

type(X_train)

# figure out how many encounters we have
numencnt = X_train.shape[0]

# choose a random patient encounter to plot
ix = random.randint(0,5000) #Try a few different index values between 0 and 4999
print('ix = {}'.format(ix))

# plot a matrix of observation values
plt.title("Patient Encounter Matrix")
plt.pcolor(np.transpose(X_train[ix,:,:]))
plt.ylabel("variable")
plt.xlabel("time/observation")
plt.ylim(0,265)
plt.colorbar()
plt.show()

  
### 2.3.1 Exercise: Padded Variable over all Patient Encounters

varnum = 227  

# TODO Step 2 Create a matrix of encounters vs time/observation
# Hint: Select along the 3rd axis
varmatrix = np.transpose(X_train[:,:,varnum]) 

# Step 3 Plot the matrix
try:
    plt.title("Variable Matrix")
    plt.pcolor(varmatrix) 
    plt.ylabel("time/observation")
    plt.xlabel("encounter")
    plt.ylim(0,600)
    plt.colorbar()
    plt.show()
except Exception as e:
    print('ERROR found: {}'.format(e))

  
## 2.4 Save a *NumPy* Arrays

# Save the prepared numpy arrays for use in other notebooks
np.save('X_train_prepared.npy',X_train,allow_pickle=False)
np.save('y_train_prepared.npy',y_train,allow_pickle=False)
np.save('X_valid_prepared.npy',X_valid,allow_pickle=False)
np.save('y_valid_prepared.npy',y_valid,allow_pickle=False)
