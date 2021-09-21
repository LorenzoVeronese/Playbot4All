
from flask import Flask, render_template, request, redirect
import os
import threading
import time
import sys
from multiprocessing import Process
import multiprocessing
from subprocess import check_output
import signal

# this creates an instance of flask running
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "/home/pi/Desktop/Playbot4All-main/Motors_code/files"


pid = 0

@app.route("/")
def index(methods = ["GETS", "POST"]):
    if request.method == "POST":
        x.terminate()
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
        if len(gcode.filename.split('.')) < 2:
            return render_template("pictureerror.html", name = name)
        if gcode.filename.split('.')[1] != 'gcode':
            return render_template("pictureerror.html", name = name)
        gcode.save(os.path.join(app.config['UPLOAD_FOLDER'], name+'.gcode'))
    else:
        return render_template("pictureerror.html", name = name)
    return render_template("picture.html", name = name)


@app.route("/deleteform")
def deleteform():
    filelist = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template("deleteform.html", filelist = filelist)


@app.route("/delete", methods=["POST"])
def delete():
    choosen = request.form.get("options")
    os.system('rm ' + '/home/pi/Desktop/Playbot4All-main/Motors_code/files/' + choosen)
    return render_template("delete.html")


@app.route("/playform")
def setplay():
    # list of file in 
    filelist = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template("playform.html", filelist = filelist)


@app.route("/play", methods = ["GET", "POST"])
def play():
    # if interrompi
    if request.method == "GET":
        # Avvio il processo del telaio (cnc program)
        os.system('rm ' + '/home/pi/Desktop/Server/processend.txt')
        fd = open('processend.txt', 'w')
        fd.write('0')
        fd.close()
        '''# Avvio il processo della camera (tracker)
        os.system('rm' + '/home/pi/Desktop/Server/cameraend.txt')
        fd = open('cameraend.txt', 'w')
        fd.write('0')
        fd.close()'''
        # Recupera il file scelto dal file
        fd = open('choosen.txt', 'r')
        choosen = fd.readline()
        fd.close()
        return render_template("play.html", choosen = choosen)
    else:
        choosen = request.form.get("options")
        fd = open('choosen.txt', 'w')
        fd.write(choosen)
        fd.close()
    
        '''
        x = threading.Thread(target=start_play)
        x.start()
        '''
        x = threading.Thread(target=start_play)
        x.start()
        '''y = threading.Thread(target=start_camera)
        y.start()'''
        '''
        fd = open('processpid.txt', 'w')
        fd.write(str(x))
        fd.close()
        '''
        return render_template("play.html", choosen = choosen)
    
    
@app.route("/playstop")
def playstop():
    os.system('rm ' + '/home/pi/Desktop/Server/processend.txt')
    fd = open('processend.txt', 'w')
    fd.write('2')
    fd.close()

    '''os.system('rm ' + '/home/pi/Desktop/Server/cameraend.txt')
    fd = open('cameraend.txt', 'w')
    fd.write('2')
    fd.close()'''


    fd = open('choosen.txt', 'r')
    choosen = fd.readline()
    fd.close()

    return render_template("playstop.html", choosen = choosen)


@app.route("/finished")
def finished():
    fd = open('processend.txt', 'w')
    fd.write('1')
    fd.close()

    '''fd = open('cameraend.txt', 'w')
    fd.write('1')
    fd.close()'''

    
    return render_template("finished.html")


def start_play():
    bashcommand = "python3 /home/pi/Desktop/Playbot4All-main/Motors_code/cnc_program.py "
    os.system(bashcommand)
'''
def start_camera():
    nashcommand = "python3 /home/pi/Desktop/Playbot4All-main/Tracker.py
    os.system(bashcommand)
'''
#if __name__ == '__main__':
#    app.run(debug = True)
