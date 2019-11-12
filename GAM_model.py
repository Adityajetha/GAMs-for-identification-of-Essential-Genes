#from pygam.datasets import wage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('frame2.csv')

Y = df['Label']
#print(Y)
X =  df[["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]]
names = ["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]
trainX, testX, trainy, testy = train_test_split(X, Y, test_size = 0.4)
from pygam import LogisticGAM, s, f
#print("allocated")
#print(trainX)
aa=s(0)
names1=["Silent"]
for i in range(1,37):
	#if(not(i==4 or i==6 or i==10 or i==11)):
	if(not(i==4 or i==6 or i==10 or i==11)):
		aa+=s(i)
		names1.append(names[i])
#print("yeh karre")
gam1 = LogisticGAM(aa)
#print(gam1.lam)
w=[]
for y in trainy:
	if(y==0):
		w.append(1)
	else:
		w.append(10)
gam1=gam1.fit(trainX,trainy,weights=w)
import numpy as np
lams = np.random.rand(10, 33) # random points on [0, 1], with shape (100, 3)
n_splines = [5,10,15,20,25]
lams = lams * 6 # shift values to -3, 3
lams=lams-3
lams = np.exp(lams)
cons= ['convex', 'concave', 'monotonic_inc', 'monotonic_dec','circular', 'none']
random = LogisticGAM(aa).gridsearch(trainX,trainy,weights=w,lam=lams,n_splines=n_splines)
random=random.gridsearch(trainX,trainy,constraints=cons)
print(random.lam)
print(random.n_splines)
print(random.constraints)
print(random.accuracy(testX,testy))

from sklearn.metrics import confusion_matrix
preds=random.predict(testX)
print(confusion_matrix(testy,preds))
for i, term in enumerate(random.terms):
    if term.isintercept:
        continue
    XX = random.generate_X_grid(term=i)
    pdep, confi = random.partial_dependence(term=i, X=XX, width=0.95)
    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(names1[i])
    plt.show()