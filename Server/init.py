from flask import Flask, render_template, request, redirect
import os

# this creates an instance of flask running
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "/home/pi/Desktop/Playbot4All-main/Motors_code/files"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pictureform")
def pictureform():
    return render_template("pictureform.html")


@app.route("/picture", methods = ["POST"])
def picture():
    name = request.form.get("name")
    
    if request.files:
        gcode = request.files["file"]
            
        # We want only gcode files
        if len(filename.split('.')) < 2:
            return render_template("pictureerror.html", name = name)
        
        if gcode.filename.split('.')[1] != 'gcode':
            return render_template("pictureerror.html", name = name)
            
        gcode.save(os.path.join(app.config['UPLOAD_FOLDER'], name+'.txt')) # .txt da sostituire con .gcode
            
    else:
        return render_template("pictureerror.html", name = name)
        
    return render_template("picture.html", name = name)


@app.route("/playform")
def setplay():
    # list of file in 
    filelist = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template("playform.html", filelist = filelist)


@app.route("/play", methods = ["POST"])
def play():
    choosen = request.form.get("options")
    
    
    
    return render_template("play.html")

#if __name__ == '__main__':
#    app.run(debug = True)
