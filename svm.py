# Importing libraries
import pandas as pd
import numpy as np
from sklearn import svm


# Read in the big dataset with flood data
data = pd.read_csv('../Data/Big/Flood.csv', index_col='Unnamed: 0')
print(data)

# Divide big datset into training and test data
train = data.loc[pd.to_datetime(data.index) <= pd.to_datetime('10/1/2018')]
print(train)
test = data.loc[pd.to_datetime(data.index) > pd.to_datetime('10/1/2018')]
print(test)


# Function to perform k nearest neighbors classification for specified features
def run_svm(x_feat, y_feat):
    print('SVM Classification for {0} based on {1}:'.format(y_feat, x_feat))

    # Train the k nearest neighbors model on training data
    model = svm.SVC()
    model.fit(train[x_feat], train[y_feat])

    # Make predictions from model for test data
    preds = model.predict(test[x_feat])
    print('{0} Flood Events Predicted'.format(sum(preds)))

    # Get actual test data results
    results = test['X_Flood'].values
    results = np.reshape(results, len(preds))
    print('{0} Flood Events Occured'.format(sum(results)))

    # Compare actual test data to predictions
    miss = sum(abs(results - preds))
    print('{0} Misclassifications'.format(miss))
    percent_succ = ((len(results) - miss) / len(results)) * 100
    print('{0} % Successfully Classified\n'.format(percent_succ))


# Run SVM classification on flood data with different input fields
# Same location input fields
run_svm(['B_Flow', 'B_Precip'], 'B_Flood')
run_svm(['X_Flow', 'X_Precip'], 'X_Flood')
run_svm(['Y_Flow', 'Y_Precip'], 'Y_Flood')
run_svm(['Z_Flow', 'Z_Precip'], 'Z_Flood')

# Other location input fields
run_svm(['A_Flow', 'B_Flow', 'B_Precip', 'D_Precip', 'E_Flow'], 'C_Flood')
run_svm(['X_Flow', 'X_Precip', 'Z_Flow', 'Z_Precip'], 'Y_Flood')
