from flask import Flask,render_template,request,send_file,send_from_directory,jsonify
from PIL import Image
from io import BytesIO

import pickle
import cv2
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import re
import base64
from keras.preprocessing.image import load_img

features=pickle.load(open('features .pkl','rb+'))

model=pickle.load(open('model.pkl','rb+'))

path=pickle.load(open('path.pkl','rb+'))

app = Flask(__name__,static_folder='static',template_folder='templates')


def img_preprocessing(img):
  npimg = np.fromstring(img,dtype=np.uint8)
  
# convert numpy array to image
  img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
  image3=cv2.resize(img,(224,224))
  image3=np.reshape(image3,(224,224,3))
  return image3

def cosinesimilarity(features,query_feature):
  cosine_distance=[]
  ii=0
  for i in features:
    dis=cosine(query_feature,i)
    cosine_distance.append([dis,ii])
    ii=ii+1
  cosine_similarity=sorted(cosine_distance)
  return cosine_similarity

def main_path(img):
  image=img_preprocessing(img)
  query_feature=model.predict(np.array([image]))
  cosine_similarity=cosinesimilarity(features,query_feature)
  paths=[path[i[1]] for i in cosine_similarity[0:10]]
  return paths

def parseImg(imgData):
    imgData="data:image/png;base64,"+imgData
    img=Image.open(BytesIO(base64.b64decode(imgData))).convert('UTF-8')

    return img



@app.route('/',methods=['POST','GET'])
def predict():
  if request.method=='POST':
    img=request.files['file'].read()
    response=main_path(img)
    print(response)
    dicts={}
    for jj in range(len(response)):
      dicts[jj]=response[jj]
    return jsonify(dicts)
  if request.method=='GET':
    return render_template('index.html')
  

if __name__=='__main__':
    app.run(debug=False,threaded=False)


