import numpy as np
import pandas as pd
x=pd.read_csv("overdoses.csv")
x["Population"] = x["Population"].str.replace(",","").astype(float)
x["Deaths"] = x["Deaths"].str.replace(",","").astype(float)
x['od']=x.Deaths/x.Population.astype(float)
diff=np.max(x['od'])-np.min(x['od'])
y=len(x['Deaths'])
simmat=[['  ']+x['Abbrev'].values.tolist()]
for i in range(50):
    list=[x['Abbrev'][i]]
    c=x['od'][i]
    normailised_list=[round(abs(c-j)/diff,3) for j in x['od']]
    maxi=np.argmax(normailised_list)
    normailised_list[maxi]=0
    normailised_list[i]=1
    list=list+normailised_list
    simmat.append(list)
outputfile=open('matrix1.csv','w')
for lis in simmat:
    print(' ',lis,file=outputfile,end='\n')
















