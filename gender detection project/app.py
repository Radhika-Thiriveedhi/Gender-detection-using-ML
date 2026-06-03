from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load accuracy
accuracy = pickle.load(open("accuracy.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        name = request.form["name"]

        vector = vectorizer.transform([name])

        result = model.predict(vector)

        prediction = result[0]

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=round(accuracy * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)