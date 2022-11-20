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


@app.route("/pred", methods=['POST'])
def model_predic_images():
    image1 = request.json['image1']
    model_name = request.json['model_name']
    print("done loading")
    prediction_1 = str(model_name).predict([image1])
    return {"prediction of given model is " :int(predicted[0])}

    #return {"compaired_result_is":int(predicted[0])}