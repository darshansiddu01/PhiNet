from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask import render_template
from werkzeug.utils import secure_filename
from models.PhiNet import *
import os
import socket
app = Flask(__name__)
api = Api(app)
Users = {}


@app.route('/Welcome')
def Welcome():
    return render_template('Welcome.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():

    user = request.form['UserId']
    if user in Users:
        image = request.files['Signature']
        return jsonify("Score: "+str('{:0.3f}'.format(F.pairwise_distance(PhiNet(PreProcess(image)), Users[user]).item())))

    else:
        return jsonify("User isn't present!")


@app.route('/register', methods=['POST'])
def register():
    user = request.form['UserId']
    image = request.files['Signature']

    if user in Users:
        return jsonify("User is already registered!")

    Users[user] = PhiNet(PreProcess(image)).data
    # image.save('/home/tug/WS/PhiNet/db/'+image.filename)
    return jsonify("User: "+str(user)+" has been successfully registered")


@app.route('/check', methods=['GET'])
def check():
    return jsonify({i: user for i, user in enumerate(Users)})


if __name__ == '__main__':
    if os.name=='nt':
        host = socket.gethostbyname(socket.gethostname())
    else:
        host = os.popen("hostname -I").read().split(" ")[0]
    PhiNet = torch.load("models/phinet_siamese_theone.stdt", map_location="cpu")['net']
    os.system("clear")
    print(" * Model has been imported....")
    print(" * PhiNet is running....")
    app.run(host=host, port="5000")
