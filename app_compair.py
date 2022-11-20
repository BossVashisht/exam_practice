from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    image = request.json['image']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}


@app.route("/compair", methods=['POST'])
def compair_images():
    image1 = request.json['image1']
    image2 = request.json['image2']
    print("done loading")
    prediction_1 = model.predict([image1])
    prediction_2 = model.predict([image2])
    if prediction_1 == prediction_2:
        predicted = [1]
    else:
        predicted = [0]
    if predicted[0] == 1:
        return {"images are same" :int(predicted[0])}
    else:
        return {"images are different" :int(predicted[0])}

    #return {"compaired_result_is":int(predicted[0])}