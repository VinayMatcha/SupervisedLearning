import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("overdoses.csv")
df['Population'] = df['Population'].replace(to_replace=r',', value='', regex=True).astype(np.int64)
df['Deaths'] = df['Deaths'].replace(to_replace=r',', value='', regex=True).astype(np.int64)
s = np.corrcoef(df['Population'], df['Deaths'])[0, 1]
print("Pearson Coefficient for the above data is ",s)
df['ODD'] = (df.Deaths / df.Population).astype(np.float64)
plt.bar(df['Abbrev'], height = df['ODD'])
plt.xlabel("State Abbrevation")
plt.ylabel("Opioid Death Density")
plt.show()
plt.savefig("Images/OddForDM")
n = len(df['Deaths'])
simm = np.zeros((n, n))
for i in range(50):
    p = df['ODD'][i]
    arr = [abs(p-x) for x in df.ODD]
    aMax =  np.max(arr)
    simm[i:,] = np.abs((arr - aMax)/aMax)
    print(df['Abbrev'][i] , " ", simm[i, 0:5])