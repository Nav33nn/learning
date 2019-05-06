from collections import defaultdict
import cv2
import sys
import os
import time
import scipy
import numpy
import operator
from array import array
import glob


outPath = '/home/naveen/Documents/hackathon/sig-na/src/test-image/'
fileList = glob.glob('/home/naveen/Documents/hackathon/sig-na/src/test-image/no-sig-pdf.jpg')

def get_tables(imgfile):
	img = cv2.imread(imgfile)
	img_area = cv2.contourArea(numpy.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]))
	factor = 1
	grayimg = cv2.resize(img, (0, 0), fx=factor, fy=factor)
	grayimg = cv2.cvtColor(grayimg, cv2.COLOR_BGR2GRAY)
	inv_gray = numpy.invert(grayimg)
	# cv2.imwrite(outPath + 'invGray.png', inv_gray)

	ret, bw_thresh = cv2.threshold(inv_gray, 127, 255, cv2.THRESH_BINARY)
# 	cv2.imwrite(outPath + 'bw_thresh.png', bw_thresh)
	horz_g = bw_thresh.copy()
	vert_g = bw_thresh.copy()

	scale = 30 # play with this

	hor_g_size, ver_g_size = tuple(hw / scale for hw in bw_thresh.shape)
	hor_g_size = int(hor_g_size)
	ver_g_size = int(ver_g_size)
	horStr = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_g_size, 1))
	erode_gh = cv2.erode(horz_g, horStr)
# 	cv2.imwrite(outPath + 'eroded_h.png', erode_gh) #SIGNATURE LINES
	dilate_gh = cv2.dilate(erode_gh, horStr)
# 	cv2.imwrite(outPath + 'dilated_h.png', dilate_gh) #SIGNATURE LINES

	verStr = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_g_size))
	erode_gv = cv2.erode(vert_g, verStr)
# 	cv2.imwrite(outPath + 'eroded_v.png', erode_gv)

	dilate_gv = cv2.dilate(erode_gv, verStr)
# 	cv2.imwrite(outPath + 'dilated_v.png', dilate_gv)

	mask_g = dilate_gh + dilate_gv
# 	cv2.imwrite(outPath + 'mask_g.png', mask_g)

	joints_g = numpy.bitwise_and(dilate_gh, dilate_gv)
# 	cv2.imwrite(outPath + 'n_joints_g.png', joints_g)
	print(len(cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)))
	_,contours, hierarchy = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	oord=cv2.findNonZero(dilate_gh)
	print(oord)

	test = img[550:700, 0: 700]
	cv2.imwrite(outPath + 'test.png', test)		


	border = 20
	tnum = 0
	tminwidth = 50
	numCountours = len(contours)
	contour_area = []
	baseImg = numpy.zeros(img.shape, dtype=numpy.uint8) + 255
# 	cv2.imwrite(outPath + 'baseImg.png', baseImg)
	for n in range(0, numCountours):
		x, y, w, h = cv2.boundingRect(contours[n])
		x = x - border
		y = y - border
		w = w + 2 * border
		h = h + 2 * border
		cv2.rectangle(joints_g, (x, y), (x + w, y + h), (255, 255, 255), 2)
		crop_img = img[y: y + h, x: x + w]
		crop_img_contour = numpy.array([[0, 0], [crop_img.shape[1], 0], [crop_img.shape[1], crop_img.shape[0]], [0, crop_img.shape[0]]])
		crop_img_area = cv2.contourArea(crop_img_contour)
		contour_area.append(crop_img_area * 100 / img_area)
		row, col, gray_code = crop_img.shape
		print(row, tminwidth, col, contour_area)
		if row > tminwidth and col > tminwidth and contour_area[n] >= 1.75:
			baseImg[y:y + crop_img.shape[0],x:x + crop_img.shape[1]] = crop_img
			tnum += 1
			print( '%s, %d, %d' % (imgfile.split('-')[0],int(imgfile.split("-")[1].split('.')[0]), tnum))
			fileList.append([imgfile.split('-')[0], [int(imgfile.split("-")[1].split('.')[0]),tnum]])

if __name__ == '__main__':
	for file in fileList:
		get_tables(file)
