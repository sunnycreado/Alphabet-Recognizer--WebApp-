
from flask import Flask,render_template,request
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


app=Flask(__name__)

classes = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                   23: 'X', 24: 'Y', 25: 'Z'}



@app.route('/',methods=['GET'])
def lol():

    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    model = load_model('font.h5')
    imagefile=request.files['imagefile']
    image_path=r"Images\\" + imagefile.filename
    imagefile.save(image_path)

    image=load_img(image_path)
#    image = img_to_array(image)
#    image=image.reshape(1,image.shape[0],image.shape[1],1)
#    img_pred = classes[np.argmax(model.predict(image))]


    new_img = image.resize((28, 28), Image.Resampling.LANCZOS)
    finalImageNotProcessed = new_img.convert('L')
    finalImageProcessed = np.array(finalImageNotProcessed).reshape((1, 28, 28, 1))
    img_pred = classes[np.argmax(model.predict(finalImageProcessed))]





    return render_template('index.html',prediction=img_pred)



if __name__=='__main__':
    app.run(port=3000,debug=True)
