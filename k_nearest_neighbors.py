# Importing libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Register converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Read in the big dataset with flood data
data = pd.read_csv('../Data/Big/Flood.csv', index_col='Unnamed: 0')
print(data)

# Divide big datset into training and test data
train = data.loc[pd.to_datetime(data.index) <= pd.to_datetime('10/1/2018')]
print(train)
test = data.loc[pd.to_datetime(data.index) > pd.to_datetime('10/1/2018')]
print(test)


# Function to perform k nearest neighbors classification for specified features
def knn(x_feat, y_feat, n):
    print('K Nearest Neighbors Classification for {0} based on {1} with {2} Neighbors:'
          .format(y_feat, x_feat, n))

    # Train the k nearest neighbors model on training data
    model = KNeighborsClassifier(n_neighbors=n)
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


# Run k nearest neighbors classification on flood data with different input fields
n = 10

# Same location input fields
knn(['B_Flow', 'B_Precip'], 'B_Flood', n)
knn(['X_Flow', 'X_Precip'], 'X_Flood', n)
knn(['Y_Flow', 'Y_Precip'], 'Y_Flood', n)
knn(['Z_Flow', 'Z_Precip'], 'Z_Flood', n)

# Other location input fields
knn(['A_Flow', 'B_Flow', 'B_Precip', 'D_Precip', 'E_Flow'], 'C_Flood', n)
knn(['X_Flow', 'X_Precip', 'Z_Flow', 'Z_Precip'], 'Y_Flood', n)
