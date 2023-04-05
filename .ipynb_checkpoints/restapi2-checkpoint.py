import os
import io
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import json
import time
import Diffuse
import GetPrompt
import jsonify
from PIL import Image
from gpt import Getprompt as gp
from img_to_3d import get_mesh_with_mc


app = Flask(__name__)
CORS(app)
@app.route('/call', methods= ['GET', 'POST'])
def call():
    if request.method == 'POST':
        get_type = request.args.get('type')
        print(get_type)
        if get_type == "Diffusion":
            prompt = request.form["text"]
            image = Diffuse.run(prompt+ "chair with out background")
            print("image_ready")
            image.save('./image.jpg')
            return send_file('./image.jpg', mimetype='image/jpeg')
        elif get_type == "3D":
            file = request.files['Image']
            file.save('./image.jpg') 
            image = Image.open('./image.jpg')
            get_mesh_with_mc.main(image)
            with open('./mesh_1b.ply', "rb") as f:
                file_data = f.read()           
            return send_file('./mesh_1b.ply')
    if request.method == 'GET':
        get_type = request.args.get('type')
        if get_type == "discriminate" :
            text = request.args.get('text')
            print(text)
            answer = gp.getAnswer(text)
            print(answer)
            #json_object = json.dumps(answer)
            return answer
    return 200  

if __name__ == '__main__':
    app.run(host='0.0.0.0',port = '5000',debug=True)