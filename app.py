from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "iris/models/model.pkl")
iris_model = joblib.load(model_path)

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
