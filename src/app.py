# Code provided here was obtained from: https://stackoverflow.com/a/54808032/4822073
# Original Author: Jacob Lawrence <https://stackoverflow.com/users/8736261/jacob-lawrence>
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
# Minor modifications made by Dharmesh Tarapore <dharmesh@cs.bu.eu>
from flask import Flask,request,jsonify, render_template, render_template_string
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

TEAM = ['Anastasiia','Michelle','Shelli','Zach','other']

app = Flask(__name__, template_folder="views")

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

    # In restrospect, we should have coordinated on the image dimensions before creating
    # our models
    image_a = image.resize((224, 224))
    image_b = image.resize((256, 256))
    
    image_a = tf.keras.preprocessing.image.img_to_array(image_a)
    image_a = image_a[...,:3]
    image_a = image_a[None,:,:,:]

    image_b = tf.keras.preprocessing.image.img_to_array(image_b)
    image_b = image_b[...,:3]
    image_b = image_b[None,:,:,:]

    face_model =  tf.keras.models.load_model('face_model')
    prob_face = 1 - face_model.predict(image_a)

    spoof_model =  tf.keras.models.load_model('spoof_model')
    prob_spoof = spoof_model.predict(image_b)

    team_model =  tf.keras.models.load_model('team_model')
    team_member_probs = team_model.predict(image_b)

    person_name = TEAM[np.argmax(team_member_probs)]

    if prob_face > 0.5 and person_name == 'other':
      return render_template('unauthorized.html')
    elif prob_face > 0.5 and prob_spoof < 0.95:
      return render_template('logged_in.html', name=person_name)
    elif prob_face > 0.5 and prob_spoof > 0.95:
      return render_template("spoofed_face.html")
    else:
      return render_template('no_face.html')

if __name__ == "__main__":
    app.debug = False
    app.run(host='0.0.0.0')
