import cv2


from scipy import misc


def crop_bottom_half(image):
    cropped_img = image[image.shape[0]/2:image.shape[0]]
    return cropped_img


def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()

imgfile = '/home/naveen/Documents/hackathon/sig-na/src/test-image/test.png'
img = cv2.imread(imgfile)
# cropped_img = crop_bottom_half(img)
# cv2.imwrite('/home/naveen/Documents/hackathon/sig-na/src/test-image/n_joints_g.png', cropped_img)


# Read the image
img = misc.imread(imgfile)
print(img.shape)
height, width,_ = img.shape

# Cut the image in half
width_cutoff = width // 2
s1 = img[:, :width_cutoff]
s2 = img[:, width_cutoff:]

s1 = cv2.cvtColor(s1, cv2.IMREAD_GRAYSCALE)
s2 = cv2.cvtColor(s2, cv2.IMREAD_GRAYSCALE)

print(count_nonblack_np(s1) )
print(count_nonblack_np(s2))
# Save each half
misc.imsave("/home/naveen/Documents/hackathon/sig-na/src/test-image/1-sig1.png", s1)
misc.imsave("/home/naveen/Documents/hackathon/sig-na/src/test-image/1-sig2.png", s2)
