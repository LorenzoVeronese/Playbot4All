from flask import Flask, render_template, request

# this creates an instance of flask running
app = Flask(__name__)

subscribers = []

@app.route('/')
def index():
    return render_template("index.html", title = title)

@app.route('/about')
def about():
    names = ["John", "Mary", "Wes", "Sally"]
    return render_template("about.html", names = names)

@app.route('/subscribe')
def subscribe():
    title = "Subscribe to my newsletter"
    return render_template("subscribe.html", title = title)

@app.route('/form', methods = ["POST"])
def form():
    email = request.form.get("email")
    subscribers.append(f"{email}")
    title = "Thanks"
    return render_template("subscribe.html", title=title, email = email, subscribers = subscribers)
