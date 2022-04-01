import cv2
import numpy as np
import os
import pickle

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

sift = cv2.xfeatures2d.SIFT_create()

# Number of centers for KMeans 
# Number of categories * 10 (100 in this case since 10 players)
k = 100

def train():

    """
    This function is used to train a model for player identification.
    It uses a bag of visual words model to classify the different images. 
    We will use SIFT algorithm to extract the keypoints of each image and create the bag of words.
    This is represented by a feature histogram.
    """
    
    img_path = 'training/'
    dico = []
    for i in range(1,201):
        # print(img_path + str(i) + ".jpg")
        img = cv2.imread(img_path + str(i) + ".jpg")
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        for d in des:
            dico.append(d)

    batch_size = np.size(os.listdir(img_path)) * 3
    # clustering the large number of descriptors using KMeans
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)

    Pkl_Filename = "kmeans_Model.pkl"  

    # saving the KMeans model after training 
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(kmeans, file)

    kmeans.verbose = False

    #Creating the image histogram. We will create a vector of k value for each image.
    #For each keypoints in an image, we will find the nearest center and increase by one its value.

    histo_list = []

    for i in range(1,201):
        img = cv2.imread(img_path + str(i) + ".jpg")
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        histo = np.zeros(k)
        nkp = np.size(kp)

        for d in des:
            idx = kmeans.predict([d])
            # Normalization 
            histo[idx] += 1/nkp 

        histo_list.append(histo)

    X = np.array(histo_list)
    Y = []

    # Manually labelling the dataset and storing it as class labels in Y 
    # 20 samples for each player in this case 
    for x in range(0,20):
        Y.append(2)
        Y.append(12)
        Y.append(3)
    
    for x in range(0,20):
        Y.append(5)
        Y.append(1)
    
    for y in range(0,20):
        Y.append(11)
    
    for y in range(0,20):
        Y.append(15)
    
    for y in range(0,20):
        Y.append(24)
    
    for y in range(0,20):
        Y.append(35)
    
    for y in range(0,20):
        Y.append(22)

    #WHITE(DUKE) -- 2,12,3,5,1
    #BLACK(UCF) --  1,15,24,35,2 --> 11,15,24,35,22

    print(len(X))
    print(len(Y))

    # Shuffling the training dataset
    X, Y = shuffle(X, Y)
    
    # initializing the Multi-layer perceptron classifier 
    mlp = MLPClassifier(verbose=True, max_iter=5000)

    # fitting the model to input X and target Y
    mlp.fit(X, Y)

    Pkl_Filename = "MLP_Model.pkl"  

    # saving the MLP model
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(mlp, file)


def predict(img):
    """
    This function is used to predict the player jersey number.
    The trained models are loaded and predict on the image in the parameter.
    Returns the jersey number of the player in the image.
    """
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)

    with open("kmeans_Model.pkl", 'rb') as file:  
        kmeans = pickle.load(file)
    
    kmeans.verbose = False

    with open("MLP_Model.pkl", 'rb') as file:  
        mlp = pickle.load(file)

    x = np.zeros(k)
    nkp = np.size(kp)
    for d in des:
        idx = kmeans.predict([d])
        x[idx] += 1/nkp

    #res = mlp.predict_proba([x])
    res = mlp.predict([x])
    prob = mlp.predict_proba([x])
    print(res)
    # print(type(res))

    print(prob)
    # print(type(prob))

    if np.max(prob) < 0.80:
        return None
    else:
        return str(res[0])

# These lines of code are for using the predict() 
img = cv2.imread("training/113.jpg")
print("b1.jpg:")
text = predict(img)
UCF = {"15":"Aubrey Dawkins","11":"B.J. Taylor","24":"Tacko Fall","35":"Collin Smith","22":"Terrell Allen","21":"Chad Brown","10":"Dayon Griffin","20":"Frank Bertz"} #BLACK
DUKE = {"1":"Zion Williamson","5":"RJ Barrett","2":"Cam Reddish","3":"Tre Jones","12":"Javin DeLaurier","14":"Jordan Goldwire","20":"Marques Bolden"} #WHITE
print(text)
print(UCF[text])
# print(DUKE[text])

# train() # Uncomment this to train the model 