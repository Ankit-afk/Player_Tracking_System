import cv2
import numpy as np
import os
import pickle

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

def detect_number(img,sift,kmeans,mlp):
    """
    This function is used for player identification.
    Takes image to perform prediction on, SIFT object used to create keypoints and descriptors 
    on the image. kmeans and mlp are the trained model objects which perform the player prediction.
    Returns the predicted jersey number associated with a player.
    """

    k = 100
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    kmeans.verbose = False

    x = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        x[idx] += 1/nkp

    res = mlp.predict([x])
    prob = mlp.predict_proba([x])

    return str(res[0])
    