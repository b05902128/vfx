import numpy as np
from scipy.ndimage import filters 
import cv2
import matplotlib.pyplot as plt 
import math
def harris(image, sigma, k): #input k for change the value easily
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
	R = delta - k*(trace**2)
	# R = delta/trace
	return R, image_x, image_y
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
	# plt.savefig("out.png")
	plt.show()

# featuere description : 將圖片獨特之處寫成數值，以供辨識比對
# (SIFT) ：
# 一個像素進行投票，梯度長度是持票數，梯度角度決定投票箱編號。 360° 等分成 8 個箱子。一個像素投票給一個箱子。
def orientation(Lx, Ly, bins, kernel_size):
	m = np.sqrt(Lx**2 + Ly**2)
	theta = np.arctan(Ly/ (Lx+1e-8))*(180 / np.pi)
	

# def descriptor():
# def feature_matching():

#grayscale_test1photo
img_gray = cv2.imread('./test_data/parrington/prtn01.jpg',cv2.IMREAD_GRAYSCALE)
img_gray = np.float32(img_gray)
cv2.imwrite('testgray1.jpg', img_gray)
print(type(img_gray), img_gray.shape)
 
 
corner_response, i_x, i_y  = harris(img_gray, 3, 0.05)
coords_x, coords_y = supression(corner_response)
# print(coor_x,coor_y)
plot_corners(img_gray, coords_x, coords_y)
