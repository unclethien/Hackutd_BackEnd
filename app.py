from flask import Flask 

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
  
  # Get image from request
  img = request.files['image'].read()
  
  # Existing OpenCV emotion detection code
  predicted_emotion = detect_emotion(img) 
  
  # Return prediction
  return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run()