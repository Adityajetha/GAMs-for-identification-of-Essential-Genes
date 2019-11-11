import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('frame2.csv')

Y = df['Label']
X = df[["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]]
names = ["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]
trainX, testX, trainy, testy = train_test_split(X, Y, test_size = 0.4)
valX, testX, valy, testy = train_test_split(testX, testy, test_size = 0.5)
from pygam import LogisticGAM, s, f
aa=s(0)
for i in range(1,37):
	aa+=s(i)
#gam1 = LogisticGAM(aa)
#print(gam1.lam)
w=[]
for y in trainy:
	if(y==0):
		w.append(1)
	else:
		w.append(10)
import numpy as np
lams = np.random.rand(10, 37) # random points on [0, 1], with shape (100, 37)
lams = lams * 6 # shift values to -3, 3
lams=lams-3
lams = np.exp(lams)
random = LogisticGAM(aa).fit(trainX,trainy,weights=w).gridsearch(trainX,trainy,weights=w,lam=lams)
print(random.lam)
#print(gam1.accuracy(testX,testy))
print(random.accuracy(testX,testy)) # get test accuracy
for i, term in enumerate(random.terms):
    if term.isintercept:
        continue
    XX = random.generate_X_grid(term=i)
    pdep, confi = random.partial_dependence(term=i, X=XX, width=0.95)
    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(names[i])
    plt.show()