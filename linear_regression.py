# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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


# Function to perform linear regression for specified input and output features
def linreg(x_feat, y_feat, title):
    # Get input and output features from training and test data
    x_train = train[x_feat]
    y_train = train[y_feat]
    x_test = test[x_feat]
    y_test = test[y_feat]

    # Run linear regression
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Make predictions from model and find mean square error
    preds = model.predict(x_test)
    mse = np.mean(np.power((np.array(y_test) - np.array(preds)), 2))


    # Plot the predictions with MSE
    title = title + '; MSE = {0}'.format(mse)
    test['Predictions'] = 0
    test['Predictions'] = preds
    plt.plot_date(pd.to_datetime(train.index), train[y_feat], fmt='-')
    plt.plot_date(pd.to_datetime(test.index), test[[y_feat, 'Predictions']], fmt='-')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(y_feat)
    plt.legend(['Training Data', 'Test Data', 'Test Prediction'])
    plt.show()

    return mse


# Perform linear regression for a variety of flood data input and output features
# Stage based on Flow and Precip at one location
site = 'Y'
title = 'Stage of {0} based on Flow and Precip'.format(site)
mse = linreg(['{0}_Flow'.format(site), '{0}_Precip'.format(site)],
             '{0}_Stage'.format(site), title)
print('{0} MSE = {1}'.format(title, mse))

# Stage based on Flow, Precip, and Flood at one location
site = 'Y'
title = 'Flow of {0} based on Flow, Precip, and Flood'.format(site)
mse = linreg(['{0}_Stage'.format(site), '{0}_Precip'.format(site), '{0}_Flood'.format(site)],
             '{0}_Flow'.format(site), title)
print('{0} MSE = {1}'.format(title, mse))

# Stage at C Crabtree location based on stages at other Crabtree locations
title = 'Stage of based on Stage of other Crabtree locations'
mse = linreg(['A_Stage', 'B_Stage', 'D_Stage', 'E_Stage'],
             'C_Stage', title)
print('{0} MSE = {1}'.format(title, mse))

# Stage at Y Urban location based on stages at other Urban locations
title = 'Stage of Y based on Stage of other Urban locations'
mse = linreg(['X_Stage', 'Z_Stage'],
             'Y_Stage', title)
print('{0} MSE = {1}'.format(title, mse))
