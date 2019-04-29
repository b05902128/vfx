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
	# plt.savefig("out.png")
	plt.show()

# feature description : 將圖片獨特之處寫成數值，以供辨識比對
# (SIFT) ：
# 一個像素進行投票，梯度長度是持票數，梯度角度決定投票箱編號。 360° 等分成 8 個箱子。一個像素投票給一個箱子。
# By assigning a consistent orientation, the keypoint descriptor can be orientation invariant.
# def orientation(Ix, Iy, bins, kernel_size):
# 	m = np.sqrt(Ix**2 + Iy**2)
# 	# 0 ~ 2pi
# 	theta = np.arctan(Iy/ (Ix+1e-8))*(180 / np.pi)
# 	theta[Ix < 0] += 180
# 	theta = (theta + 360) % 360
# 	# 4*4+8bin的descirptor是最好的，因此每个关键点将会产生128维的特征向量。
# 	binsize = 360. / bins
# 	theta_bin = (theta + binsize / 2) // int(binsize) % bins
# 	print(theta)
# 	print("binsize = ", binsize)
# 	print("theta_bin = ", theta_bin)
# Simplest solution :
def description(image, coor_x, coor_y, win):
	descriptor = np.zeros((500,(2*win+1)*(2*win+1)))
	for i in range(500):
		x = int(coor_x[i])
		y = int(coor_y[i])
		print(x)
		print(y)
		print(i)
		one_dim = image[y-win:y+win+1, x-win:x+win+1].flatten()
		print(one_dim.shape)
		descriptor[i] = one_dim
	return descriptor






#grayscale_test1photo
img_gray = cv2.imread('./parrington/prtn00.jpg',cv2.IMREAD_GRAYSCALE)
img_gray = np.float32(img_gray)
cv2.imwrite('testgray.jpg', img_gray)
print(type(img_gray), img_gray.shape)
 
 
corner_response  = harris(img_gray, 3, 0.05)
coords_x, coords_y = supression(corner_response)
# print(coor_x,coor_y)
plot_corners(img_gray, coords_x, coords_y)
des = description(img_gray, coords_x, coords_y, 3)
print(des.shape)