import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from sklearn.neighbors import KNeighborsClassifier
import cv2


#################### Functions ##########################
# calculate multifractal spectrum
def multifractal_spectrum(data):
    # Calculate the power spectrum for each feature vector in data
    power_spectra = [np.abs(fft(vector)) ** 2 for vector in data]
    # Shift the zero-frequency component to the center for each power spectrum
    power_spectra = [fftshift(spectrum) for spectrum in power_spectra]
    return power_spectra


# build image using predictions
def rebuildImg(predictions):
    predcounter = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            orgImgPx = maskimg[i, j]
            if not (orgImgPx[0] == 0):
                if (predictions[predcounter] == 0):
                    imgAfterMask[i, j] = [255, 0, 0]
                else:
                    imgAfterMask[i, j] = [0, 0, 255]
                predcounter += 1


#######################################################


# image to process
originalimg = cv2.imread("eyeSection1.jpg")
# mask of image showing vessel segmentation
maskimg = cv2.imread("es1rip.png")
# training image with veins and arteries
trainingImg = cv2.imread("mappedvessels.png")

# apply mask
imgAfterMask = cv2.bitwise_and(originalimg, maskimg)
original = imgAfterMask

X_test = []
feature_vectors = []
labels = []
rows, cols, _ = imgAfterMask.shape
background = 0
for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        k = trainingImg[i, j]
        orgImgPx = maskimg[i, j]
        # generate test dataset from original image excluding background
        if not (orgImgPx[0] == 0):
            featureArray = np.concatenate((originalimg[i, j], originalimg[i - 1, j], originalimg[i + 1, j],
                                           originalimg[i, j - 1], originalimg[i, j + 1], originalimg[i - 1, j - 1],
                                           originalimg[i - 1, j + 1], originalimg[i + 1, j - 1],
                                           originalimg[i + 1, j + 1]), axis=None)
            X_test.append(featureArray)
            background += 1
        if k[0] == 0:
            background += 1
        # generate training dataset from original image using matches from delineated image
        elif k[0] == 255:
            featureArray = np.concatenate((originalimg[i, j], originalimg[i - 1, j], originalimg[i + 1, j],
                                           originalimg[i, j - 1], originalimg[i, j + 1], originalimg[i - 1, j - 1],
                                           originalimg[i - 1, j + 1], originalimg[i + 1, j - 1],
                                           originalimg[i + 1, j + 1]), axis=None)
            feature_vectors.append(featureArray)
            labels.append(0)  # artery label
        else:
            featureArray = np.concatenate((originalimg[i, j], originalimg[i - 1, j], originalimg[i + 1, j],
                                           originalimg[i, j - 1], originalimg[i, j + 1], originalimg[i - 1, j - 1],
                                           originalimg[i - 1, j + 1], originalimg[i + 1, j - 1],
                                           originalimg[i + 1, j + 1]), axis=None)
            feature_vectors.append(featureArray)
            labels.append(1)  # vein label

# X_train = power_spectra
y_train = labels
X_train = multifractal_spectrum(feature_vectors)

# Classify the new data using K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier()

# Train the classifier on the training data
print("Fitting training data")
knn.fit(X_train, y_train)

print("Classifying test data")
X_test = multifractal_spectrum(X_test)
# Use the trained classifier to predict the labels of the testing data
predictions = knn.predict(X_test)

rebuildImg(predictions)

# show the classifcation image
cv2.imshow('Classified image', imgAfterMask)
cv2.waitKey(0)
cv2.destroyAllWindows()
