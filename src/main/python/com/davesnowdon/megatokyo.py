'''
Created on Feb 19, 2012

@author: dns
'''
import os
import fnmatch

import PythonMagick

import cv

MEGATOKYO_DIR = '/opt/data/library/fiction/megatokyo'
NUM_BINS = 64

def get_src_images():
    imagefiles = []
    for f in sorted(os.listdir(MEGATOKYO_DIR)):
        if fnmatch.fnmatch(f, '*.gif') or fnmatch.fnmatch(f, '*.jpg'):
            imagefiles.append(f)
    return imagefiles

def get_png_images():
    imagefiles = []
    for f in sorted(os.listdir(MEGATOKYO_DIR)):
        if fnmatch.fnmatch(f, '*.png'):
            imagefiles.append(f)
    return imagefiles
    
def make_histogram(imagefile):
    col = cv.LoadImageM(imagefile)
    gray = cv.CreateImage(cv.GetSize(col), cv.IPL_DEPTH_8U, 1)
    cv.CvtColor(col, gray, cv.CV_RGB2GRAY)
    
    hist = cv.CreateHist([NUM_BINS], cv.CV_HIST_ARRAY)
    cv.CalcHist(gray, hist)
    return hist
    
ims = get_images()
for im in ims:
    h = make_histogram(MEGATOKYO_DIR +'/'+ im)
    print im, " : ", h
    