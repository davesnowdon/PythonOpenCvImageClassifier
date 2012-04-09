'''
Created on Feb 19, 2012

@author: dns
'''
import os
import sys
import fnmatch
import getopt

import PythonMagick

import cv

MEGATOKYO_DIR = '/opt/data/library/fiction/megatokyo'
COMIC_IMAGES_SUBDIR = 'comic'
UNWANTED_IMAGES_SUBDIR = 'dontread'
NUM_BINS = 64

def get_src_images(basedir):
    imagefiles = []
    for f in sorted(os.listdir(basedir)):
        if fnmatch.fnmatch(f, '*.gif') or fnmatch.fnmatch(f, '*.jpg'):
            imagefiles.append(f)
    return imagefiles

def get_png_images(basedir):
    imagefiles = []
    for f in sorted(os.listdir(basedir)):
        if fnmatch.fnmatch(f, '*.png'):
            imagefiles.append(f)
    return imagefiles

def convert_file_name_to_png(filename):
    return os.path.splitext(filename)[0] + '.png'

def convert_image_to_png(basedir, src):
    image = PythonMagick.Image(basedir + '/' + src)
    dest  = convert_file_name_to_png(src)
    # sometimes fails with a runtime error: Magick: tRNS chunk has out-of-range samples for bit_depth
    image.write(basedir + '/' + dest)

# Get images after converting to PNG format if necessary
def get_images(basedir):
    gif_and_jpg = get_src_images(basedir)
    png = get_png_images(basedir)
    images = []
    for f in gif_and_jpg:
        fpng = convert_file_name_to_png(f)
        if fpng not in png:
            print "Converting: "+f
            convert_image_to_png(basedir, f)
        images.append(fpng)
    return images
    
def make_histogram(imagefile):
    col = cv.LoadImageM(imagefile)
    gray = cv.CreateImage(cv.GetSize(col), cv.IPL_DEPTH_8U, 1)
    cv.CvtColor(col, gray, cv.CV_RGB2GRAY)
    
    hist = cv.CreateHist([NUM_BINS], cv.CV_HIST_ARRAY)
    cv.CalcHist(gray, hist)
    return hist


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

# Ideas for defining main() from http://www.artima.com/weblogs/viewpost.jsp?thread=4829
def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
        
        if 0 == len(args):
            raise Usage("Missing base path")
        basedir = args[0].strip()
        print "Base dir = " + basedir
        
        ims = get_images(basedir)
        for im in ims:
            h = make_histogram(basedir +'/'+ im)
            print im, " : ", h
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    
if __name__ == "__main__":
    sys.exit(main())
