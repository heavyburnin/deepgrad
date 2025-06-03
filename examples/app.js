# app.py
from flask import Flask, render_template, request, jsonify
import dill
from tensor import Tensor
import os

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = dill.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('pixels')
    if not data or len(data) != 784:
        return jsonify({'error': 'Invalid input'}), 400

    x = Tensor(data, requires_grad=False, shape=(1, 784))
    pred = model(x)
    logits = list(pred.data)
    predicted_class = max(range(10), key=lambda i: logits[i])
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

