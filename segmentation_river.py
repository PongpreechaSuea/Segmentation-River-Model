from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from config import *
from PIL import Image, ImageDraw
import cv2
import numpy as np

class SegmentationRiver():
    def __init__(self):
        self.model = YOLO(MODEL)
        self.cache = CACHE
        self.conf = CONF
        self.color= COLOR_AREA
        self.alpha= ALPHA
        self.min_distance = MIN_DISTANCE
        
    def predict_on_image(self,img):
        result = self.model(img, conf=self.conf,verbose=False)[0]
        masks = result.masks.data.cpu().numpy()
        masks = np.moveaxis(masks, 0, -1)

        masks = scale_image(masks, result.masks.orig_shape)
        masks = np.moveaxis(masks, -1, 0)
        return masks

    def overlay(self,image, mask, resize=None):
        color = self.color[::-1]
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()

        if resize is not None:
            image = cv2.resize(image.transpose(1, 2, 0), resize)
            image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

        image_combined = cv2.addWeighted(image, 1 - self.alpha, image_overlay, self.alpha, 0)
        return image_combined

    def show(self,photo):
        img = Image.open(photo)
        dim = np.array(img).shape
        if dim[0] > LIMIT_PIXEL:
          sub_image = (dim[0] // SIZE_SUB)
          img = img.resize((int(dim[1]/sub_image) , int(dim[0]/sub_image) ), resample=1)

        elif dim[1] > LIMIT_PIXEL:
          sub_image = (dim[1] // SIZE_SUB)
          img = img.resize((int(dim[1]/sub_image) , int(dim[0]/sub_image) ), resample=1)

        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        masks = self.predict_on_image(img)

        image_with_masks = np.copy(img)
        for mask_i in masks:
            image_with_masks = self.overlay(image_with_masks, mask_i)

        return image_with_masks

    def predict(self,photo):
        try :
            photo = cv2.imread(photo)
        except:
            pass
        result = self.model(photo, conf=self.conf , verbose=False)[0]
        green_pixels = result[0].masks.xy[0][::self.min_distance]
        for x,y in green_pixels:
            cv2.circle(photo, (int(x), int(y)), 5, COLOR_MARK, -1)
        int32_array = np.array(green_pixels, dtype=np.int16)
        
        return  int32_array , photo


if __name__ in "__main__":
    sr = SegmentationRiver()
    photo = "./data/dataset/91.jpg"

    pixels , image = sr.predict(photo)
    print(f"[ จำนวนตำแหน่ง pixel จัดกลุ่มมาได้ ] : {len(pixels)}")

    ##############################################

    # image = sr.show(photo)

    cv2.imshow("output",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
