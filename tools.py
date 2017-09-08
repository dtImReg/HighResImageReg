import numpy as np
import cv2

importCheck = 'Imported correctly'

def preProc(imgOne = None,kernelSize = 0, sigmaX = 0):
    for i in range(1):
        imgOne = cv2.bilateralFilter(imgOne,kernelSize,sigmaX,sigmaX)
        #imgOne = cv2.GaussianBlur(imgOne,(kernelSize-20,kernelSize-20),sigmaX-10) #31x41 works good for HigQ
        #imgOne = cv2.medianBlur(imgOne,11)
        imgOne = cv2.equalizeHist(imgOne)
        print 'Pre-proc complete. Completed histo equalization and blurring'
    return imgOne


def preProcLowRes(imgOne = None,kernelSize = 0, sigmaX = 0):
    for i in range(1):
        imgOne = cv2.bilateralFilter(imgOne,kernelSize,sigmaX,sigmaX)
        #imgOne = cv2.GaussianBlur(imgOne,(kernelSize-20,kernelSize-20),sigmaX-10) #31x41 works good for HigQ
        imgOne = cv2.equalizeHist(imgOne)
        print 'Pre-proc complete. Completed histo equalization and blurring'
    return imgOne

def downSample(img = None,factor = 0L):
    if img.ndim == 2 :
        height,width = img.shape
    else:
        height,width,_ = img.shape
        
    dwnsImg = cv2.resize(img,(int(factor*width), int(factor*height)), interpolation = cv2.INTER_CUBIC)
    return dwnsImg
    
    
#====================================================Image Pre-Processing Code ===============================#
# =================================================== Blob Detector ========================================== #
def detectTissue(img = None, threshold = 0 , kernelSize = 41 , erosionIteration = 20):
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    noiseKernel = np.ones((3,3),np.uint8)
    
    
    grayImage = cv2.GaussianBlur(grayImage,(kernelSize,kernelSize),75)
    grayImage = cv2.equalizeHist(grayImage)
    
    ret, thresh = cv2.threshold(grayImage,threshold,255,cv2.THRESH_BINARY_INV) #105
    
    #noise removal
    
    blobs = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,noiseKernel,iterations =2)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,noiseKernel,iterations =2)
    
    blobs = cv2.erode(blobs,noiseKernel, iterations =20)
    
    return blobs

def extractTissue (img = None ,blobs = None, backgroundColour = (255,255,255)) :
    
    _,contours, hierarchy = cv2.findContours(blobs, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    height,width,channels = img.shape
    
    array = []
    for i in range(len(contours)):
        array.append(cv2.contourArea(contours[i]))
        #print cv2.contourArea(contours[i])    
    test = [i for i in range(len(contours)) if array[i]== max(array)]
    idx = test[0] # The index of the contour that surrounds your object
    
    
    white_image = np.zeros((height,width,channels), np.uint8)
    white_image[:] = backgroundColour

    #img1 =  white_image #white_image
    #img2 = blobs #img

    # Create the ROI of the subject image
    roi = white_image[0:height, 0:width ]

    mask = np.zeros((height,width), np.uint8)

    cv2.drawContours(mask, contours, idx, (255,255,255), -1) # Draw filled contour in mask

    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)



    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img,img,mask = mask)

    # Put extracted bit in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    white_image[0:height, 0:width ] = dst
    
    return white_image,contours,idx    
    

def detectAndExtractTissue(img = None, threshold = 0 , kernelSize = 41 , backgroundColour = (255, 255, 255), erosionIteration = 20):
    grayImage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    noiseKernel = np.ones((3,3),np.uint8)
    
    
    grayImage = cv2.GaussianBlur(grayImage,(kernelSize,kernelSize),75)
    grayImage = cv2.equalizeHist(grayImage)
    
    ret, thresh = cv2.threshold(grayImage,threshold,255,cv2.THRESH_BINARY_INV) #105
    
    #noise removal
    
    blobs = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,noiseKernel,iterations =2)
    blobs = cv2.morphologyEx(blobs,cv2.MORPH_CLOSE,noiseKernel,iterations =2)
    
    blobs = cv2.erode(blobs,noiseKernel, iterations =20)
    
    _,contours, hierarchy = cv2.findContours(blobs, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    height,width,channels = img.shape
    
    array = []
    for i in range(len(contours)):
        array.append(cv2.contourArea(contours[i]))
        #print cv2.contourArea(contours[i])    
    test = [i for i in range(len(contours)) if array[i]== max(array)]
    idx = test[0] # The index of the contour that surrounds your object
    
    
    white_image = np.zeros((height,width,channels), np.uint8)
    white_image[:] = backgroundColour

    #img1 =  white_image #white_image
    #img2 = blobs #img

    # Create the ROI of the subject image
    roi = white_image[0:height, 0:width ]

    mask = np.zeros((height,width), np.uint8)

    cv2.drawContours(mask, contours, idx, (255,255,255), -1) # Draw filled contour in mask

    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)



    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img,img,mask = mask)

    # Put extracted bit in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    white_image[0:height, 0:width ] = dst
    
    return white_image,contours,idx


def boundRectTissue(img = None, contours= None, idx = -1, borderSize = 0):
    
    x,y,w,h = cv2.boundingRect(contours[idx])
    res = img[y:y+h,x:x+w]
    
    return res,x,y,w,h
   
def regImageReposition(img = None, bgImage = None, (x,y,w,h) = None, whiteBg = True):

    height,width,channels = bgImage.shape
    
    if whiteBg == True:
        white_image = np.zeros((height,width), np.uint8)
        white_image[:] = 255
    else:
        white_image = bgImage

    white_image[y:y+h,x:x+w] = img
        
    return white_image