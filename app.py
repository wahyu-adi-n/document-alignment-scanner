import cv2
import numpy as np
import os, shutil
from flask import Flask, render_template, request, redirect, url_for

class DocumenScanner():

    def __init___(self, image):
        self.image = self.readImage(image)

    def readImage(self, image):
        image = cv2.imread(image) 
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def resizeImage(self, image):
        dim_limit = int(1080)
        max_dim = max(image.shape)
        if max_dim > dim_limit:
            resize_scale = dim_limit / max_dim
            image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
        return image

    def copyImage(self, image):
        return image.copy()

    def morphoOperation(self, image):
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=3)
        return image
    
    def grabCut(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)
        backModel = np.zeros((1, 65), np.float64)
        foreModel = np.zeros((1, 65), np.float64)
        rect = (20, 20, image.shape[1] - 20, image.shape[0] - 20)
        cv2.grabCut(image, mask, rect, backModel, foreModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        image = image * mask2[:, :, np.newaxis]
        return image
    
    def edgeDetection(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blur = cv2.GaussianBlur(image_gray, (11, 11), 0)
        canny = cv2.Canny(image_blur, 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))   
        return canny
    
    def findCountour(self, canny):
        img, contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        return page
    
    def orderPoints(self, pts):
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype('int').tolist()
    
    def findDestination(self, pts):
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        destinationCorners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return self.orderPoints(destinationCorners)
    
    def cornerDetection(self, page, image):
        if len(page) == 0:
            return image
        for c in page:
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break
        corners = sorted(np.concatenate(corners).tolist())
        corners = self.orderPoints(corners)
        destination_corners = self.findDestination(corners)
        return corners, destination_corners
    
    def getHomography(self, image, corners, destination_corners):
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        final = cv2.warpPerspective(image, M, (destination_corners[2][0], \
                    destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        return final
    
    def saveImage(self, path, image):
        cv2.imwrite(path, image)

def pipeline(image, result):
    DS = DocumenScanner()
    image = DS.readImage(image)
    image_resize = DS.resizeImage(image)
    original = DS.copyImage(image_resize)
    morph = DS.morphoOperation(image_resize)
    grabcut = DS.grabCut(morph)
    canny = DS.edgeDetection(grabcut)
    contours = DS.findCountour(canny)
    corners, destination_corners = DS.cornerDetection(contours, original)
    aligned = DS.getHomography(original, corners, destination_corners)
    DS.saveImage(result, aligned)

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def index():
    if(request.method == 'GET'):
        return render_template('index.html')
    elif(request.method == 'POST'):
        original = 'static/original.jpg'
        result = 'static/result.jpg'
        img_original = request.files['original']
        img_original.save(original)
        pipeline(original, result)
        return redirect(url_for('result'))

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = os.getenv('PORT', 5000)
    app.debug = True
    app.run(host='0.0.0.0', port=port)
    shutil.rmtree('static')