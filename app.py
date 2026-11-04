from flask import Flask, render_template, request
from src.predict import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    article = ""
    if request.method == "POST":
        article = request.form.get("article", "").strip()
        try:
            result = predict_news(article)
        except ValueError as error:
            result = {"error": str(error)}
    return render_template("index.html", result=result, article=article)

if __name__ == "__main__":
    app.run(debug=True)
