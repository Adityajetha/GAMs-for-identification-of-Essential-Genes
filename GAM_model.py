#from pygam.datasets import wage
import pandas as pd
import matplotlib.pyplot as plt
#from keras.utils import normalize
from sklearn.model_selection import train_test_split
df = pd.read_csv('frame2.csv')

Y = df['Label']
#print(Y)
X =  df[["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]]
#X=normalize(X)
#print(X)
names = ["Silent","Missense_Mutation","3'Flank","Splice_Site","Frame_Shift_Ins","Nonsense_Mutation","In_Frame_Ins","5'UTR","RNA","Intron","Frame_Shift_Del","In_Frame_Del","3'UTR","5'Flank","Splice_Region","Translation_Start_Site","Nonstop_Mutation","Missense/Silent","Non-Silent/Silent","SNP","AC","AG","AT","CA","CG","CT","GA","GC","GT","TA","TC","TG","Minimum Expression Value","Maximum Expression Value","Copy Number -1","Copy Number 0","Copy Number 1"]
trainX, testX, trainy, testy = train_test_split(X, Y, test_size = 0.9)
valX, testX, valy, testy = train_test_split(testX, testy, test_size = 0.9)
from pygam import LogisticGAM, s, f
print("allocated")
#print(trainX)
aa=s(0)
names1=["Silent"]
for i in range(1,37):
	if(not(i==4 or i==6 or i==10 or i==11)):
		aa+=s(i)
		names1.append(names[i])
print("yeh karre")
gam1 = LogisticGAM(aa)
print(gam1.lam)
w=[]
for y in trainy:
	if(y==0):
		w.append(1)
	else:
		w.append(10)
gam1=gam1.fit(trainX,trainy,weights=w)
import numpy as np


lams = np.random.rand(10, 33) # random points on [0, 1], with shape (100, 3)

lams = lams * 6 # shift values to -3, 3
#print(lams)
lams=lams-3
#print(lams)
lams = np.exp(lams)
#print(lams)
random = LogisticGAM(aa).fit(trainX,trainy,weights=w).gridsearch(trainX,trainy,weights=w,lam=lams)
print(random.lam)
#random = LogisticGAM(s(0) + s(1) + f(2)).fit(trainX, trainy).gridsearch(trainX, trainy, lam=lams)
print(gam1.accuracy(testX,testy))
print(random.accuracy(testX,testy))
for i, term in enumerate(gam1.terms):
    if term.isintercept:
        continue
    print(i)
    XX = gam1.generate_X_grid(term=i)
    pdep, confi = gam1.partial_dependence(term=i, X=XX, width=0.95)
    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    #plt.scatter(trainX[i], trainy, facecolor='gray', edgecolors='none')
    plt.title(names1[i])
    plt.show()
for i, term in enumerate(random.terms):
    if term.isintercept:
        continue
    XX = random.generate_X_grid(term=i)
    pdep, confi = random.partial_dependence(term=i, X=XX, width=0.95)
    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    #plt.scatter(trainX[i], trainy, facecolor='gray', edgecolors='none')
    plt.title(names1[i])
    plt.show()
#gam.summary()
#print(gam.summary())