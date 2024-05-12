from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)



model = load_model ('coco_disease.h5')


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        f = request.files['file']
        file_path = 'uploads/' + f.filename
        f.save(file_path)
        
       
        processed_image = preprocess_image(file_path)

      
        predictions = model.predict(processed_image)
        class_names =  ['CCI_Caterpillars','CCI_Leaflets' ,'WCLWD_DryingofLeaflets','WCLWD_Flaccidity', 'WCLWD_Yellowing'] # Replace with your class names
        predicted_class = class_names[np.argmax(predictions)]
        if predicted_class=='CCI_Caterpillars':
            predicted_class='CCI Caterpillars'
        elif predicted_class=='CCI_Leaflets':
            predicted_class='CCI Leaflets'
        elif predicted_class=='WCLWD_DryingofLeaflets':
            predicted_class='WCLWD Drying of leaflets'
        elif predicted_class=='WCLWD_Flaccidity':
            predicted_class='WCLWD Flaccidity'
        elif predicted_class=='WCLWD_Yellowing':
            predicted_class='WCLWD Yellowing'

        return render_template('result.html', prediction=predicted_class, image_file_name=f.filename)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
