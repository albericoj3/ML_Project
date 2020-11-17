from keras.models import load_model
from pandas import np
import IPython.display as ipd

from ML_Project.Preprocessing import classes, x_val, y_val

model=load_model('Project_Model.hdf5')

#Predict Audio
def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


import random
index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))