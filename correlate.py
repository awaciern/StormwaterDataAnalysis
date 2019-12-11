# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read in the big dataset with flood data
data = pd.read_csv('../Data/Big/Flood.csv', index_col='Unnamed: 0')
print(data)

# Calculate correlation matrix and save it to a csv
cols = [col for col in data]
corr_matrix = pd.DataFrame(np.corrcoef([data[col] for col in data]),
                           columns=cols, index=cols)
print(corr_matrix)
corr_matrix.to_csv('../Data/Corr/Correlation_Matrix.csv')

# Plot the correlation matrix
plt.matshow(corr_matrix)
plt.xticks(range(len(corr_matrix.columns)),corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix for Flood Data Across All Locations')
plt.show()
