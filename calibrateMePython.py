import numpy as np
import cv2
import glob
import sys
from termcolor import colored, cprint

    
class cameras:
    def __init__(self, path, nRows, nCols, dimension, time):
        self.path = path
        self.nRows = nRows
        self.nCols = nCols
        self.dimension = dimension
        self.time = time
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.images = glob.glob(self.path + '*.jpg')
        self.badframes = []
        self.goodframes = []
    
    def criteria(self):
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.dimension, 0.001)
    
    def objp(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nRows*self.nCols,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nCols,0:self.nRows].T.reshape(-1,2)
        return objp
    
    def calibrate(self):
        if len(self.images) < 1:
            #print("can't find enough images if any")
            cprint("\n ---> Can't find enough images if any", 'red', attrs=['bold', 'underline', 'blink'])
            sys.exit()
        for fname in self.images:
            #print("reading frame {}".format(fname))
            cprint(("reading frame " + fname), 'white', attrs=['dark'])
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #print (gray.shape[::-1]) #print image size
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nCols,self.nRows),None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(self.objp())
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria())
                self.imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.nCols,self.nRows), corners,ret)
                cv2.imshow('img',img)
                cv2.waitKey(self.time)
                self.goodframes.append(fname)
            else:
                #print(fname, "is a bad frame")
                cprint(fname + " is a bad frame", 'red')
                self.badframes.append(fname)
        cv2.destroyAllWindows()
        if len(self.goodframes) == 0:
            #print ("Chessboars size doesn't match")
            cprint("\n ---> Chessboars size doesn't match", 'red', attrs=['bold', 'underline', 'blink'])
            sys.exit()
        elif len(self.goodframes) < 10:
            #print ("need more good frames")
            cprint("\n ---> need more good frames", 'red', attrs=['bold', 'underline', 'blink'])
            sys.exit()
        else:
            print ("starting to calibrate mono camera")
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            print ("finished calibrating")
            return self.mtx, self.dist, self.rvecs, self.tvecs, self.objpoints, self.imgpoints
            
    def undistort(self, image):
        self.calibrate()
        img = cv2.imread( self.path + image + '.jpg')
        print (self.path + image + '.jpg' )
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
        #Crop
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(self.path + image + 'Dist.png',dst)

    def errors(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
            Total_errors = str(mean_error/len(self.objpoints))
            #print (error)
        #print ( "total error: {}".format(mean_error/len(self.objpoints)) )
        cprint("total error: "+ Total_errors, 'yellow', attrs=['bold'])

def mono_calibrate():
    LeftCam.calibrate()
    LeftCam.errors()
    RightCam.calibrate()
    RightCam.errors() 

def stereo_calibrate():
    mono_calibrate()
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    
    imageSize = (cv2.imread(LeftCam.images[0]).shape[1::-1])
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    if (len(LeftCam.imgpoints)) != (len(RightCam.imgpoints)):
        #print("delete bad frames from both cameras and try again")
        cprint("\n ---> Delete bad frames from both cameras and try again.", 'red', attrs=['bold', 'underline', 'blink'])
    else:
        print ("Starting to calibrate stereo camera.")
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate( LeftCam.objpoints, LeftCam.imgpoints, RightCam.imgpoints, LeftCam.mtx, LeftCam.dist, RightCam.mtx, RightCam.dist, imageSize, criteria=stereocalib_criteria, flags=flags)
        #print('Intrinsic_mtx_1', M1)
        #print('dist_1', d1)
        #print('Intrinsic_mtx_2', M2)
        #print('dist_2', d2)
        #print('R', R)
        #print('T', T)
        #print('E', E)
        #print('F', F)
        #print ('calibration was successful')
        cprint("\n ---> Stereo Calibration was successful.", 'grey', 'on_green',  attrs=['bold', 'underline', 'blink'])
    
    
DIR = sys.argv[1]
#DIR = "" #___manual directory override___#
LeftCam = cameras(DIR + "/LEFT/", 7, 9, 20, 50)  #(Name of the cameera, chessboard ver points, her points, chessboard size, wait time)
RightCam = cameras(DIR + "/RIGHT/", 7, 9, 20, 50)
#Left.undistort('left6')
stereo_calibrate()
