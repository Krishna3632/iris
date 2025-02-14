from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the model
iris_model = joblib.load("iris/models/model.pkl")

@app.route('/', methods=['GET', 'POST'])
def iris():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        ans = iris_model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        ans_name=""
        if ans[0]==0:
            ans_name="Setosa"
        elif ans[0]==1:
            ans_name="Versicolor"
        else:
            ans_name="Virginica"
        return render_template("index.html", prediction=ans_name)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)