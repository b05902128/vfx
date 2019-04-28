import numpy as np
from scipy.ndimage import filters 
import cv2
import matplotlib.pyplot as plt 
def harris(image, sigma):
	# derivatives
	image_x = np.zeros(image.shape)
	image_y = np.zeros(image.shape)
	# compute components of the Harris matrix
	# image_x, image_y = np.gradient(image)
	filters.gaussian_filter(image, (sigma, sigma), (0, 1), image_x)
	filters.gaussian_filter(image, (sigma, sigma), (1, 0), image_y)
	xx = image_x * image_x
	xy = image_x * image_y
	yy = image_y * image_y
	Ixx = filters.gaussian_filter(xx, sigma)
	Ixy = filters.gaussian_filter(xy, sigma)
	Iyy = filters.gaussian_filter(yy, sigma)
	# 指 {\displaystyle \mathbf {A} } \mathbf{A}的主對角線（從左上方至右下方的對角線）上各個元素的總和
	delta = Ixx * Iyy - Ixy * Ixy
	trace = Ixx + Iyy +1e-10
	# R = detM - k(trace(M)^2)
	# 0.04 < k < 0.06
	R = delta - 0.05*(trace**2)
	# R = delta/trace
	return R
def supression(R,win=7):
	R[  :win ,  :] = 0
	R[-win:  ,  :] = 0
	R[  :  ,  :win] = 0
	R[  :  ,-win:] = 0
	# median  = np.median(R)
	# print(median)
	# print(np.max(R))
	# print(np.min(R))
	# print(R)
	minimum = np.min(R)
	print(minimum)
	h = R.shape[0]
	w = R.shape[1]
	print(h,w)
	x = []
	y = []
	maximum = []
	for i in range(500):
		max = np.argmax(R)
		maximum.append(max)
		maxh = max//w
		maxw = max%w
		y.append(maxh)
		x.append(maxw)
		R[maxh-win:maxh+win+1,maxw-win:maxw+win+1] = minimum
	return (x,y)
 
# plot corner
def plot_corners(img, coor_x, coor_y): 
	# fig = figure(figsize=(15, 8))
	plt.imshow(img, cmap = 'gray')
	plt.plot(coor_x, coor_y, 'r*', markersize=1)
	plt.axis('off')
	plt.savefig("out.png")
 
# def plot_corners(img, coords): 
#     # fig = figure(figsize=(15, 8))
#     plt.imshow(img, cmap = 'gray')
#     plt.plot(coords[:, 0], coords[:, 1], 'r*', markersize=5)
#     plt.axis('off')
#     plt.show()
 
#grayscale_test1photo
img_gray = cv2.imread('./parrington/prtn01.jpg',cv2.IMREAD_GRAYSCALE)
img_gray = np.float32(img_gray)
cv2.imwrite('testgray.jpg', img_gray)
print(type(img_gray), img_gray.shape)
 
 
corner_response  = harris(img_gray, 3)
coords_x, coords_y = supression(corner_response)
# print(coor_x,coor_y)
plot_corners(img_gray, coords_x, coords_y)
