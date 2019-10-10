#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:21:22 2019

@author: wenyi
"""

import numpy as np
from matplotlib import pylab as plt
import matplotlib.cm as cm
import imageio
from sklearn.linear_model import LogisticRegression

#from scipy import misc
#%matplotlib inline


train_labels, train_data, test_labels, test_data = [], [], [], []

for line in open('./faces/train.txt'):
#    im = misc.imread(line.strip().split()[0])
    im = imageio.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])

for line in open('./faces/test.txt'):
    im = imageio.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])


train_data, train_labels,  = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)
test_data, test_labels =  np.array(test_data, dtype=float), np.array(test_labels, dtype=int)

plt.imshow(train_data[1].reshape(50,50), cmap = cm.Greys_r)
plt.show()

plt.imshow(test_data[0].reshape(50,50), cmap = cm.Greys_r)
plt.show()

# 1.c | compute the average face
average_face = train_data.mean(axis=0)
plt.imshow(average_face.reshape(50,50), cmap = cm.Greys_r)
plt.show()

# 1.d | subtract average face
new_train_data = train_data - average_face
new_test_data = test_data - average_face

plt.imshow(new_train_data[1].reshape(50,50), cmap = cm.Greys_r)
plt.show()

plt.imshow(new_test_data[0].reshape(50,50), cmap = cm.Greys_r)
plt.show()


# 1.e | perform SVD and compute eigenface (VT)
U, s, VT = np.linalg.svd(new_train_data)
VT = np.array(VT, dtype=float)

plt.figure(figsize=(10, 5))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    ax.imshow(VT[i].reshape(50,50), cmap = cm.Greys_r)
    
    
# 1.f | r-rank approx error
errors = []
for r in range(1,201):
    # U[:,: r]  Σ[: r,: r]  VT [: r,:]
    rank_r = np.dot(np.dot(U[:,:r],np.diag(s[:r])),VT[:r,:])
    errors.append(np.linalg.norm(new_train_data - rank_r))

rs = range(1, 201)

plt.figure(figsize=(8, 5))
plt.plot(rs, errors)

plt.xlabel("r")
plt.ylabel("rank-r approximation error")
plt.show()

# 1.g | generate eigenface feature matrix
def eigenface_feature(matrix, r):
    # to get F, multiply X to the transpose of first r rows of VT   (VT[: r,:])
    
    result = np.dot(matrix, VT[:r,:].T)
    return result

# extract training and test features for r = 10
eigenface_feature(new_train_data, 10)
eigenface_feature(new_test_data, 10)



# 1.f | ovr Logistic Regression
logreg = LogisticRegression()
accuracy = []

for r in range(1, 201):
    print(r)
    F = eigenface_feature(new_train_data,r)
    FT = eigenface_feature(new_test_data,r)
    logreg.fit(F, train_labels)
    accuracy.append(logreg.score(FT, test_labels))

# plot accuracy vs r
plt.figure(figsize=(8, 5))
plt.plot(rs, accuracy)
plt.xlabel("r")
plt.ylabel("accuracy")
plt.show()

# max_accuracy = accuracy
# max_accuracy.sort