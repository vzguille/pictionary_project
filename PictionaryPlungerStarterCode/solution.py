# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model


model = load_model('model_goodfix_1000-02.keras')

NS = 28
classes_text = ['accessory',
 'cats',
 'construction',
 'fruit',
 'instrument',
 'one_liner',
 'plant',
 'shape',
 'sport',
 'terrain',
 'tool',
 'vehicle',
 'weapon',
 'weather',
 'writing_utensil']

class Solution:
    def __init__(self):
        pass

    # this is a signal that a new drawing is about to be sent
    def new_case(self):
        self.x_org = []
        self.y_org = []

        print(model.summary())
        pass

    # given a stroke, return a string of your guess
    def guess(self, x, y, fix = True) -> str:
        if fix == True:
            stroke_x = x
            stroke_y = y
            x_stroke = stroke_x[:]
            y_stroke = stroke_y[:]
            len_stroke = len(stroke_x)
            
            xn = []
            yn = []
            for i in range(len_stroke - 1):
                if (x_stroke[i+1] - x_stroke[i]) >= (y_stroke[i+1] - y_stroke[i]):
                    ran = int(x_stroke[i+1] - x_stroke[i])
                    x_save = x_stroke[i] + np.linspace(0, ran, np.abs(ran))
                    otro_ran = np.clip(y_stroke[i+1] - y_stroke[i], 1, 1000)
                    y_save = (((x_save - x_stroke[i] )/ran)*(otro_ran)) + y_stroke[i]
                    y_save = list(np.round(y_save, 0 ).astype(int))
                        
                else:
                    ran = int(y_stroke[i+1] - y_stroke[i])
                    y_save = y_stroke[i] + np.linspace(0, ran, np.abs(ran))
                    otro_ran = np.clip(x_stroke[i+1] - x_stroke[i], 1, 1000)
                    x_save = (((y_save - y_stroke[i] )/ran)*(otro_ran)) + x_stroke[i]
                    x_save = list(np.round(x_save, 0 ).astype(int))
                    
                
                
                xn.extend(x_save)
                yn.extend(y_save)
                
            x_save = [x_stroke[-1]]
            y_save = [y_stroke[-1]]
            
            xn.extend(x_save)
            yn.extend(y_save)
            
            stroke_x = xn[:]
            stroke_y = yn[:]
            
            self.x_org.extend(list(stroke_x))
            self.y_org.extend(list(stroke_y))
            
#                 print(stroke_x)
#                 print(stroke[0])
#                 break
        else:

            self.x_org.extend(x[:])
            self.y_org.extend(y[:])


        mms_x = MinMaxScaler()
        mms_x.fit(np.array(self.x_org).reshape(-1,1))

        mms_y = MinMaxScaler()
        mms_y.fit(np.array(self.y_org).reshape(-1,1))


        x_final = np.round(mms_x.transform(np.array(self.x_org).reshape(-1,1)).ravel()*(NS-1)).astype(int)
        y_final = np.round(mms_y.transform(np.array(self.y_org).reshape(-1,1)).ravel()*(NS-1)).astype(int)
    
        image_i = np.zeros((NS,NS),dtype = float)
        image_i[x_final, y_final] = 1.0
        image_i = image_i.reshape(1,image_i.shape[0],image_i.shape[1])
        image_i = tf.convert_to_tensor(image_i)
        res = classes_text[np.argmax(model.predict(image_i))]
        return res
        pass

    def add_score(self, score: int):
        print(score)
        pass
