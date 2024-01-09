import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from camera_multi import Camera
from ultralytics import YOLO
import cv2

import streamlit as st
import modin.pandas as pd
import ray
ray.init()

model = YOLO('./config/segv3.pt')
app = FastAPI()

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
   return templates.TemplateResponse('index.html', {"request": request})

def gen(camera):
    frame , img = camera.get_frame()
    results = model(source=img , conf=0.3 , device="cpu" , verbose=False)
    try:
        green_pixels = results[0].masks.xy[0][::20]
        for x,y in green_pixels:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imencode('.jpg', img)[1].tobytes()
    except:
        pass
    yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.get('/video_feed', response_class=HTMLResponse)
async def video_feed():
    return  StreamingResponse(gen(Camera()),media_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)