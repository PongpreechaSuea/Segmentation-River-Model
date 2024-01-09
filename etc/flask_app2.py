from flask import Flask, render_template, Response
from segmentation_river import SegmentationRiver
from camera_multi import Camera
import os
import cv2

app = Flask(__name__)
sr = SegmentationRiver()

def gen(camera):
    while True:
        frame , img = camera.get_frame()
        try:
            green_pixels , img = sr.predict(img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
        except:
            pass
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route("/")
def read_root():
    return {"model": "Segmentation river Augmented Reality" ,
            "confidence" : f"{int(sr.conf*100)}%",
            }

# /conf/
@app.get("/conf/<var>")
def conf(var):
    if var == '':
      var = 50
    var = float(var)
    if var <= 1 :
      sr.conf = var
    else:
      sr.conf = var / 100
    return {"model": "Segmentation river Augmented Reality" ,
            "confidence" : f"{int(sr.conf*100)}%" ,
            }

# /predicts/
@app.get("/predicts/<image>")
async def predicts(image):
    if image == None:
        return {"error": "Please provide context information."}
    pixels , _ = sr.predict(image)
    pixels_list = pixels.tolist()
    filename = os.path.basename(image)
    name = filename.split(".")[-2]
    extension = filename.split(".")[-1]
    return {"model": "Segmentation river Augmented Reality" ,
            "confidence" : f"{int(sr.conf*100)}%",
            "results" : { "inputs" : image ,
                          "name" : name ,
                          "extension" : extension.upper() ,
                          "pixels_count" : len(pixels_list) ,
                          "position_xy" : pixels_list } ,
            }

@app.route('/template')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8052, debug=True, threaded=True)