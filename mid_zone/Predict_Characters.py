import os
import cv2
import copy
import numpy as np
from keras.models import load_model
from scipy import stats

import tensorflow as tf

Model = load_model(os.path.join('Model', 'best_val_loss.hdf5'))

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: 'अ', 37: 'आ', 38: 'इ', 39: 'ई', 40: 'उ', 41: 'ऊ', 42: 'ऋ', 43: 'ए', 44: 'ऐ', 45: 'ओ', 46: 'औ', 47: 'अं', 48: 'अ:'}

def Predict(Characters):
    Predictions = []

    for Characters in Characters:
        Prediction = []

        for Character in Characters:
            gray = cv2.cvtColor(Character, cv2.COLOR_BGR2GRAY)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            for i in range(morph.shape[0]):
                for j in range(morph.shape[1]):
                    if not morph[i][j]:
                        morph[i][j] = 1
            
            div = gray / morph
            gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_AREA)

            x = np.array([thresh]).reshape(-1, 32, 32, 1) / 255.0
            s=Model.predict(x)
            y = np.array([np.argmax(s)])
            print(y)
            Prediction.append(Label_Dict[int(y[0])])

        Predictions.append(copy.deepcopy(Prediction))

    return Predictions