# We are importing the libraries
import numpy as np 
import pandas as pd 
from PIL import Image,ImageOps 
import tensorflow as tf 
import uvicorn 
from fastapi import FastAPI, File, UploadFile 
from listOfFlowers import CLASSES

# Importing our previously trained_model
new_model = tf.keras.models.load_model('Efficienet_B7_model.h5')

# We are creating the app object
app = FastAPI()

@app.get('/')
# Printing a Welcome message
def welcome():
	return {'message':'Welcome to the Flower Classifier API'}

@app.post('/predict')
def predict(file:UploadFile = File(...)):
	img = Image.open(file.filename)
	img = img.resize((512,512))
	img = np.array(img)
	img = img/255.0
	img = img.reshape(1,512,512,3)
	y_pred = new_model.predict(img)
	predict_index = np.argmax(y_pred,axis=1)[0]

	flower = CLASSES[predict_index]

	return {'Type of flower':f'{flower}'}


if __name__ == "__main__":
	uvicorn.run(app,host = '127.0.0.1',port=8000)


#uvicorn flower_classifier:app --reload