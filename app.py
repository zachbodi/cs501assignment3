# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask,request,jsonify, render_template
from io import BytesIO
from PIL import Image
import tensorflow as tf
import base64

app = Flask(__name__, template_folder="src/views")

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
    msg = base64.b64decode(request.form['video_feed'][22:])
    buf = BytesIO(msg)
    image = Image.open(buf)

    face_image = image.resize((224, 224))
    spoof_image = image.resize((256, 256))

    face_image = tf.keras.preprocessing.image.img_to_array(face_image)
    face_image = face_image[...,:3]
    face_image = face_image[None,:,:,:]

    spoof_image = tf.keras.preprocessing.image.img_to_array(spoof_image)
    spoof_image = spoof_image[...,:3]
    spoof_image = spoof_image[None,:,:,:]

    face_model =  tf.keras.models.load_model('src/face_model')
    spoof_model =  tf.keras.models.load_model('src/spoof_model')

    prob_no_face = face_model.predict(face_image)
    prob_spoof = spoof_model.predict(spoof_image)
    
    if prob_no_face < 0.5 and prob_spoof < 0.8:
        return render_template("logged_in.html")
    elif prob_no_face < 0.5:
        return render_template("spoofed_face.html")
    return render_template('no_face.html')

if __name__ == "__main__":
    app.run(debug=True)
