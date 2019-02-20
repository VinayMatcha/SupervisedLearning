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
# plt.show()
# plt.savefig("Images/OddForDM")
diff = np.max(df['ODD']) - np.min(df['ODD'])
n = len(df['Deaths'])
s = set(df.ODD)
print(len(s), " hhhihib")
simm = [[" "] + (df['Abbrev'].values.tolist())]
for i in range(50):
    p = df['ODD'][i]
    arr1 = [abs(p-x)/diff for x in df.ODD]
    aMax = np.max(arr1)
    aar2 = []
    # for j in arr1:
    #     if j == aMax:
    #         aar2.append(0)
    #     else:
    #         aar2.append(j)
    # arr1 = aar2
    arr1[i] = 1
    arr1 = [df['Abbrev'][i]] + arr1
    simm.append(arr1)
    #print(simm[i])

for i in range(1,50):
    for j in range(1, 50):
        if(simm[i][j] != simm[j][i]):
            print(df['Abbrev'][i] + " " + df['Abbrev'][j])