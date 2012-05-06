'''
Webcomic scanner

Usage: megatokyo BASE_DIR BASE_URL
BASE_DIR = where saved images are stored on disk
BASE_URL = where to look for comic images

@author: Dave Snowdon
'''
import os
import sys
import fnmatch
import getopt
import urllib2

import PythonMagick

import cv
import opencv.ml as ml
import numpy as np

COMIC_IMAGES_SUBDIR = 'comic'
UNWANTED_IMAGES_SUBDIR = 'dontread'
NUM_BINS = 64
BUF_LEN = 65536

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

def most_recent(imagelist):
    max = 0
    for i in imagelist:
        b = int(os.path.splitext(i)[0])
        if b > max:
            max = b
    return max

def download_file(base_url, fname, outputdir):
    try:
        r = urllib2.Request(url=base_url+fname)
        response = urllib2.urlopen(r)

        f = open(outputdir + "/" + fname, "wb")
        while True:
            buf = response.read(BUF_LEN)
            if 0 == len(buf):
                break
            f.write(buf)
        
        f.close()
        return fname
    except urllib2.HTTPError:
        return None

# Look for images to load after the most recent one on disk
def get_new_images(base_url, outputdir, most_recent):
    new_images = []
    imagenum = most_recent + 1

    while True:
        i = download_file(base_url, str(imagenum)+'.gif', outputdir)
        if i is None:
            i = download_file(base_url, str(imagenum)+'.jpg', outputdir)
            if i is None:
                break
        convert_image_to_png(outputdir, i)
        new_images.append(i)
        imagenum = imagenum + 1
    return new_images
    
def make_histogram(imagefile):
    col = cv.LoadImageM(imagefile)
    gray = cv.CreateImage(cv.GetSize(col), cv.IPL_DEPTH_8U, 1)
    cv.CvtColor(col, gray, cv.CV_RGB2GRAY)
    
    hist = cv.CreateHist([NUM_BINS], cv.CV_HIST_ARRAY, [[0,255]], 1)
    cv.CalcHist([gray], hist)
    cv.NormalizeHist(hist, 1.0)
    return hist

def make_histograms(basedir, images):
    hmap = {}
    for im in images:
        fn = basedir +'/'+im
        hmap[im] = make_histogram(fn)
    return hmap

def make_link(basedir, subdir, name):
    os.chdir(basedir)
    os.symlink('../'+name, subdir+'/'+name)


class SVM:
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = None
    
    def make_keys(self, keys):
        self.inames = {} # map key to index
        self.namei = []  # map index to key
        c = 0
        for k in keys:
            self.inames[k] = c
            self.namei.append(k)
            c = c + 1
    
    def key_to_index(self, k):
        return self.inames[k]
    
    def index_to_key(self, i):
        return self.namei[i]
        
    def train(self, samples, responses):
        #setting algorithm parameters
        params = ml.CvSVMParams()
        params.kernel_type = ml.CvSVM.LINEAR
        params.svm_type = ml.CvSVM.C_SVC
        params.C = 1
        
        self.model = ml.CvSVM()
        s = cv.CreateMat(1, NUM_BINS, cv.CV_32FC1)
        cv.Set(s, 1.0)
        v = cv.CreateMat(1, NUM_BINS, cv.CV_32FC1)
        cv.Set(v, 1.0)
        self.model.train(samples, responses, s, v,  params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])


# train a support vector machine to recognize the images based on histograms
def learn(classified, histograms):
    clf = SVM()
    clf.make_keys(classified.keys())
    
    total_samples = 0
    for c in classified.keys():
        cim = classified[c]
        total_samples = total_samples + len(cim)
        
    samples = cv.CreateMat(total_samples, NUM_BINS, cv.CV_32FC1)
    responses = cv.CreateMat(total_samples, 1, cv.CV_32FC1)
    i = 0
    for c in classified.keys():
        cim = classified[c]
        idx = clf.key_to_index(c)
        for im in cim:
            hist = histograms[im]
            for j in range(NUM_BINS):
                samples[i, j] = cv.QueryHistValue_1D(hist, j)
            responses[i, 0] = idx
        i = i + 1
        

    clf.train(samples, responses)
    return clf

def classify(basedir, category_names):
    all_images = get_images(basedir)
    all_classified_images = []
    classified = {}
    for c in category_names:
        pimg = get_png_images(basedir+'/'+c)
        classified[c] = pimg
        for im in pimg:
            all_classified_images.append(im)
    
    # now need to find the images which are not classified yet
    unclassified = []
    for i in all_images:
        if i not in all_classified_images:
            unclassified.append(i)
    
    # make histograms of all images
    hmap = make_histograms(basedir, all_images)
    clf = learn(classified, hmap)
    
    usamples = []
    for u in unclassified:
        usamples.append(hmap[u])
    
    predictions = clf.predict(usamples)
    print str(predictions)
    

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
        baseurl = args[1].strip()
        print "Base url = " + baseurl
        
        ims = get_images(basedir)
        most_recent_on_disk_comic = most_recent(ims)
        print "Most recent comic on disk is: "+ str(most_recent_on_disk_comic)
        
        #new_ims = get_new_images(baseurl, basedir, most_recent_on_disk_comic)
        #print "Downloaded "+str(len(new_ims))+" images"
        
        classify(basedir, ['comic', 'dontread'])
        
#        for im in ims:
#            h = make_histogram(basedir +'/'+ im)
#            hv = []
#            for i in range(NUM_BINS):
#                hv.append(cv.QueryHistValue_1D(h, i))
#            print im, " : ", str(hv)
#            break
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    
if __name__ == "__main__":
    sys.exit(main())
