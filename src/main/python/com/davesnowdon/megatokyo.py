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
from PyML import VectorDataSet
from PyML import SVM


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

# train a support vector machine to recognize the images based on histograms
def learn(classified, histograms):
    clf = SVM()

    
    total_samples = 0
    for c in classified.keys():
        cim = classified[c]
        total_samples = total_samples + len(cim)
        
    samples = []
    labels = []
    for c in classified.keys():
        cim = classified[c]
        for im in cim:
            hist = histograms[im]
            row = []
            for j in range(NUM_BINS):
                row.append(cv.QueryHistValue_1D(hist, j))
            samples.append(row)
            labels.append(c)

    data = VectorDataSet(samples, L=labels)
    print str(data)
    clf.train(data)
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
        hist = hmap[u]
        row = []
        for j in range(NUM_BINS):
            row.append(cv.QueryHistValue_1D(hist, j))
        usamples.append(row)
    
    data = VectorDataSet(usamples, patternID=unclassified)
    results = clf.test(data)
    
    patterns = results.getPatternID()
    labels = results.getPredictedLabels()

    # make map of image name to predicted label
    lmap = {}
    for i in range(len(patterns)):
        lmap[patterns[i]] = labels[i]
    return lmap
    

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
        
        new_ims = get_new_images(baseurl, basedir, most_recent_on_disk_comic)
        print "Downloaded "+str(len(new_ims))+" images"
        
        if len(new_ims) == 0:
            print "No new images to classify"
        else:
            labels = classify(basedir, ['comic', 'dontread'])
            print "Results for new images\n"
            for n in new_ims:
                p = convert_file_name_to_png(n)
                print p + " -> " + labels[p]

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    
if __name__ == "__main__":
    sys.exit(main())
