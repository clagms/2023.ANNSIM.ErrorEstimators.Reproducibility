import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# filename
filepath = '../../datasets/'
filename = '2022-01-24 Synch Emin.xlsx'
# read Excel file
df = pd.read_excel(filepath+filename)

# Split Params to columns
df['Params'] = df['Params'].str.replace(' ','', regex=True)
df[['m1', 'c1', 'd1','m2','c2','d2','cc','dc']] = df['Params'].str.split(',', expand=True)
df = df.drop(columns='Params')
#convert them to numeric
df['m1'] = df['m1'].str.replace('(', '', regex=True)
df['dc'] = df['dc'].str.replace(')', '', regex=True)
for column in df.columns[1:]:
	df[column] = pd.to_numeric(df[column], downcast="float")

print(df.columns)

df_mask = df['Guess+Best Count'] < 1.0
badResults_df = df[df_mask]

corr = badResults_df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

plt.tight_layout()

plt.show()