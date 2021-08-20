from flask import Flask, render_template, request

# this creates an instance of flask running
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/pictureform")
def pictureform():
    return render_template("pictureform.html")

@app.route("/picture", methods = ["POST"])
def picture():
    name = request.form.get("name")
    return render_template("picture.html", name=name)


#if __name__ == '__main__':
#    app.run(host = '0.0.0.0', port = 5000)
