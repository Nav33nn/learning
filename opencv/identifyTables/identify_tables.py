
from collections import defaultdict
import cv2
import sys
import os
import time
import scipy
import numpy
import operator
from array import array

outPath = '/home/research_user/rajiv/files/images/out/'
readPath = '/home/research_user/rajiv/files/images/'
pnoFileId=''
argDict={'-rp':'readPath','-op':'outPath','-of':'pnoFileId' }#,'-c':'inpFileDir','-dm':'modelFileDir','-h':'headerLog','-o':'fname','-B':'customer','-t':'modelMode','-ct':'concatMode','-de':'delim','-i':'randId'}
argPairs={sys.argv[i]:sys.argv[i+1] for i in range(1,len(sys.argv[1:]),2)}
for i in argPairs:
        try:
            #print('ARG  DICT',i)
            globals().update({argDict[i]:argPairs[i]})
            #for j in argDict:
            #   print(str(argDict[j]),locals()[argDict[j]])
        except KeyError as ex:
            #if ex==i:
            raise KeyError ('INVALID ARGUMENT BRO!!> '+str(ex))
if outPath[-1]!='/':
    outPath+='/'
if readPath[-1]!='/':
    readPath+='/'
fileList = []
fileDictDef = defaultdict(list)


def get_tables(imgfile):
    # print 'processing %s' % (imgfile)
    img = cv2.imread(imgfile)
    img_area = cv2.contourArea(
        numpy.array([[0, 0], [img.shape[1], 0], [
                    img.shape[1], img.shape[0]], [0, img.shape[0]]]))
    # print img.shape
    # print "image area =", img_area

    factor = 1  # 0.5
    grayimg = cv2.resize(img, (0, 0), fx=factor, fy=factor)

    grayimg = cv2.cvtColor(grayimg, cv2.COLOR_BGR2GRAY)
    inv_gray = numpy.invert(grayimg)
    # cv2.imwrite(outpath + 'invGray.png', inv_gray)

    ret, bw_thresh = cv2.threshold(inv_gray, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(outpath + 'bw_thresh.png', bw_thresh)

    horz_g = bw_thresh.copy()
    vert_g = bw_thresh.copy()

    scale = 30  # play with this

    # (bw_thresh.shape) / scale
    hor_g_size, ver_g_size = tuple(hw / scale for hw in bw_thresh.shape)

    horStr = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_g_size, 1))
    erode_gh = cv2.erode(horz_g, horStr)
    # cv2.imwrite(outpath + 'eroded_h.png', erode_gh)
    dilate_gh = cv2.dilate(erode_gh, horStr)
    # cv2.imwrite(outpath + 'dilated_h.png', dilate_gh)

    # ver_g_size = int(vert_g.shape[0] / scale)
    verStr = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_g_size))
    erode_gv = cv2.erode(vert_g, verStr)
    # cv2.imwrite(outpath + 'eroded_v.png', erode_gv)
    dilate_gv = cv2.dilate(erode_gv, verStr)
    # cv2.imwrite(outpath + 'dilated_v.png', dilate_gv)

    mask_g = dilate_gh + dilate_gv
    # cv2.imwrite(outpath + 'mask_g.png', mask_g)

    joints_g = numpy.bitwise_and(dilate_gh, dilate_gv)
    # cv2.imwrite(outpath + 'n_joints_g.png', joints_g)

    contours, hierarchy = cv2.findContours(
        mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    border = 20
    tnum = 0
    tminwidth = 50
#    avgconht = 0
    numCountours = len(contours)
    contour_area = []
    baseImg = numpy.zeros(img.shape, dtype=numpy.uint8) + 255
    # cv2.imshow("whitepage", baseImg)
    # cv2.waitKey()

    # print 'Number of contours', len(contours)

    for n in xrange(0, numCountours):
        x, y, w, h = cv2.boundingRect(contours[n])
        x = x - border
        y = y - border
        w = w + 2 * border
        h = h + 2 * border
        cv2.rectangle(joints_g, (x, y), (x + w, y + h), (255, 255, 255), 2)
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        crop_img = img[y: y + h, x: x + w]
        crop_img_contour = numpy.array([[0, 0], [crop_img.shape[1], 0], [
            crop_img.shape[1], crop_img.shape[0]], [0, crop_img.shape[0]]])
        crop_img_area = cv2.contourArea(crop_img_contour)

        contour_area.append(crop_img_area * 100 / img_area)

        row, col, gray_code = crop_img.shape
        if row > tminwidth and col > tminwidth and contour_area[n] >= 1.75:

            # Take new image and add that to baseImg
            baseImg[y:y + crop_img.shape[0],
                    x:x + crop_img.shape[1]] = crop_img
            tnum += 1

    # print 'found %d contours, and %d tables' % (len(contours), tnum)
    print '%s, %d, %d' % (imgfile.split('-')[0],
                         int(imgfile.split("-")[1].split('.')[0]), tnum)
    fileList.append(
        [imgfile.split('-')[0], [int(imgfile.split("-")[1].split('.')[0]),
                                 tnum]])

    if len(contours) >= 3 and tnum == 0:
        # print 'possible open top table'
        minX, minY, ign = img.shape
        maxX = 0
        maxY = 0
        for n in xrange(0, numCountours):
            x, y, w, h = cv2.boundingRect(contours[n])
            if minX > x:
                minX = x
            if minY > y:
                minY = y
            if maxX < x + w:
                maxX = x + w
            if maxY < y + h:
                maxY = y + h

        crop_img = img[minY - border: maxY + border,
                       minX - border: maxX + border]

        cv2.rectangle(crop_img, (border, border),
                      (maxX - minX + border, maxY - minY + border),
                      (0, 0, 0,), 2)
        baseImg[minY - border: maxY + border,
                minX - border: maxX + border] = crop_img

#    print "writing %s?%s" % (outpath, imgfile[:-3])
    cv2.imwrite(outpath + imgfile[:-4] + '-tables' + '.png', baseImg)

# End of get_tables

# imgfile = '13099'


def get_table(someFile):
    # print "%s" % someFile
    # img = cv2.imread(someFile)
    # print img.shape
    print '%s, %d' % (someFile.split('-')[0],
                      int(someFile.split("-")[1].split('.')[0]) + 1)


#os.chdir(readPath)
if pnoFileId=='':
    fileList=sorted(glob.glob(readPath+'/*.png'))
else:
    pnos=open(pnoFileId,'rb').read().replace('\r','').split('\n')
    fileList=[readPath+i+'.png' for i in pnos]#sorted(glob.glob(readPath+'/*.png'))]
i = 0
tList = []

for aFile in fileList:
#    print 'processing %s' % aFile
    print os.path.isdir(aFile)
    if os.path.isdir(aFile) is False:
        i += 1
        t1 = time.clock()
        get_tables(aFile)
        tList.append([aFile, time.clock() - t1])
    else:
        print '\nFound dir %s. Ignoring' % aFile

#    if i >= 10:
#        break

print '\nTotal of %d images' % i

totTime = 0
for timing in tList:
     totTime += timing[1]
#     print timing

print "Total Time:", totTime
cv2.destroyAllWindows()

for pNo, listItem in fileList:
    fileDictDef[pNo].append(listItem)

print fileDictDef

cv2.destroyAllWindows()

