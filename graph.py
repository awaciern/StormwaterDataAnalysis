# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# Register converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Read in the big dataset with flood data
data = pd.read_csv('../Data/Big/Flood.csv', index_col='Unnamed: 0')
print(data)

# # Plot creek stage data with flood markers
# # Site A
# plt.plot_date(pd.to_datetime(data.index), data['A_Stage'], fmt='b-', xdate=True)
# a_flood = data.loc[data['A_Flood'] == 1]
# plt.plot_date(pd.to_datetime(a_flood.index), a_flood['A_Stage'], fmt='ro', xdate=True)
#
# # Site B
# plt.plot_date(pd.to_datetime(data.index), data['B_Stage'], fmt='g-', xdate=True)
# b_flood = data.loc[data['B_Flood'] == 1]
# plt.plot_date(pd.to_datetime(b_flood.index), b_flood['B_Stage'], fmt='r*', xdate=True)
#
# # # Site C
# # plt.plot_date(pd.to_datetime(data.index), data['C_Stage'], fmt='y-', xdate=True)
# # c_flood = data.loc[data['C_Flood'] == 1]
# # plt.plot_date(pd.to_datetime(c_flood.index), c_flood['C_Stage'], fmt='r+', xdate=True)
#
# Site D
plt.plot_date(pd.to_datetime(data.index), data['C_Stage'], fmt='m-', xdate=True)
d_flood = data.loc[data['D_Flood'] == 1]
plt.plot_date(pd.to_datetime(d_flood.index), d_flood['D_Stage'], fmt='rx', xdate=True)
#
# # Site E
# # plt.plot_date(pd.to_datetime(data.index), data['C_Stage'], fmt='c-', xdate=True)
# # e_flood = data.loc[data['E_Flood'] == 1]
# # plt.plot_date(pd.to_datetime(e_flood.index), e_flood['E_Stage'], fmt='r^', xdate=True)
#
# # Labeling
# plt.xlabel('Date')
# plt.ylabel('Stage (ft)')
# plt.legend(['A_Stage', 'A_Flood', 'B_Stage', 'B_Flood', 'D_Stage', 'D_Flood'])
# plt.title('Crabtree Creek Stage Levels with Flood Markers')
# plt.show()

# Plot urban stage data with flood markers
# Site X
plt.plot_date(pd.to_datetime(data.index), data['X_Stage'], fmt='b-', xdate=True)
x_flood = data.loc[data['X_Flood'] == 1]
plt.plot_date(pd.to_datetime(x_flood.index), x_flood['X_Stage'], fmt='ro', xdate=True)

# # Site Y
# plt.plot_date(pd.to_datetime(data.index), data['Y_Stage'], fmt='g-', xdate=True)
# y_flood = data.loc[data['Y_Flood'] == 1]
# plt.plot_date(pd.to_datetime(y_flood.index), y_flood['Y_Stage'], fmt='r*', xdate=True)
#
# # Site Z
# plt.plot_date(pd.to_datetime(data.index), data['Z_Stage'], fmt='m-', xdate=True)
# z_flood = data.loc[data['Z_Flood'] == 1]
# plt.plot_date(pd.to_datetime(z_flood.index), z_flood['Z_Stage'], fmt='rx', xdate=True)

# Labeling
plt.xlabel('Date')
plt.ylabel('Stage (ft)')
plt.legend(['X_Stage', 'X_Flood', 'Y_Stage', 'Y_Flood', 'Z_Stage', 'Z_Flood'])
plt.title('Urban Raleigh Creeks Stage Levels with Flood Markers')
plt.show()
