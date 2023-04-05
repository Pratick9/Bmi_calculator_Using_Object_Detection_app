from flask import Flask,request,jsonify
import pickle
import base64
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from tensorflow.keras.preprocessing import image
HWA = pickle.load(open('HWA.pkl', 'rb'))

app = Flask(__name__)
@app.route("/")
def showHomePage():
	return "This is home page"

@app.route("/predict", methods=["POST"])
def predict():
	testing=[]
	img_path='test202.jpg'
	image_path = request.form.get('sample')
	img = image.load_img(img_path,target_size=(400,400,3))
	img = image.img_to_array(img)
	img = img/255
	testing.append(img)
	testi=np.array(testing)
	testi=testi/255
	nsamples, nx, ny,m = testi.shape
	check = testi.reshape((nsamples,nx*ny*m))
	prob1=HWA.predict(check)
	inch=prob1[0,0]*12
	bmi=(703*prob1[0,1])/inch
	print(image_path)
	if bmi<18.5:
		return jsonify({'BMI':"Underweight",'Age':str(prob1[0,2])})
	elif bmi>18.5 and bmi<24.9:
		return jsonify({'BMI': "Normal",'Age':str(prob1[0,2])})
	elif bmi>25 and bmi < 29.9:
		return jsonify({'BMI': "Overweight",'Age':str(prob1[0,2])})
	else:
		return jsonify({'BMI': "Obesity",'Age':str(prob1[0,2])})





if __name__ == "__main__":
	app.run(host="0.0.0.0")
