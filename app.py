from flask import Flask, render_template, request
import joblib
import requests
import os

app = Flask(__name__)

# External URL where your model is stored (Google Drive, AWS S3, etc.)
MODEL_URL = "https://huggingface.co/krishnash16/Iris/resolve/main/model.pkl"
MODEL_PATH = "model.pkl"

# Function to download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded!")

download_model()  # Ensure model is available

# Load the model
iris_model = joblib.load(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def iris():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['sepal_length'])
            sepal_width = float(request.form['sepal_width'])
            petal_length = float(request.form['petal_length'])
            petal_width = float(request.form['petal_width'])
            ans = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

            ans_name = ["Setosa", "Versicolor", "Virginica"][ans[0]]
            return render_template("index.html", prediction=ans_name)
        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run()
