'''
Takes an initial list of manually classified images and sets up symlinks

@author: Dave Snowdon
'''

import sys
import getopt

from megatokyo import Usage, make_link

dontread = [ '0031', '0045', '0065', '0076', '0082', '0086', '0093', '0104', '0130', '0170', '0186', '0191', '0227', '0228', '0242', '0257', '0265', '0279', '0302', '0315', '0320', '0328', '0361', '0388', '0411' ]

def make_categories(negative):
    positive = []
    ineg = []
    for i in negative:
        ineg.append(int(i))
    
    for i in range(1, max(ineg)):
        tmp = str(i)
        pstr = "00000"[0:(4-len(tmp))] + tmp
        if pstr not in negative:
            positive.append(pstr)
    
    return { 'comic' : positive, 'dontread' : negative}

def make_links(basedir, categories):
    for k in categories.keys():
        vs = categories[k]
        for v in vs:
            make_link(basedir, k, v+".png")

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
        make_links(basedir, make_categories(dontread))
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2
    
if __name__ == "__main__":
    sys.exit(main()) 