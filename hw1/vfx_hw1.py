import os
import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
# w_fn
def weight_fn(pixel_value):
	zmin = float(0)
	zmax = float(255)
	zmiddle = (zmin + zmax) / 2
	r = (zmax - zmin) / 2
	pixel_value = abs(pixel_value - zmiddle)/100
	pixel_value = -(pixel_value - r)/128
	return pixel_value

def response_fn(Z,b,l,w_fn):
	# Z(ij) = pixel value of pixel location number i in image j
	# b[j] = log delta t for image j
	# l = smooth lambda
	#w(z)  = the weight funtion value for pixel value Z
	zmin = 0
	zmax = 255
	n = 255
	z1 = Z.shape[0] 
	z2 = Z.shape[1]
	A = np.zeros((z1*z2+n, n+1+z1), dtype=np.float32)
	B = np.zeros((z1*z2+n, 1),dtype= np.float32)
	# Include the data-fitting equations
	k = 0
	for i in range(z1):
		for j in range(z2):
			w = w_fn(Z[i, j])
			A[k, Z[i, j]] = w
			A[k, n+1+i] = -w
			B[k, 0] = w * b[j]
			k+=1
	# %% Include the smoothness equations
	for i in range(1, n):
		wi = w_fn(i)
		A[k, i-1] = l * wi
		A[k, i] = -2*l*wi
		A[k, i+1] = l * wi
		k +=1
	# %% Fix the curve by setting its middle value to 0
	A[k, (zmax - zmin)//2] = 1
	# %% Solve the system using SVD

	invA = np.linalg.pinv(A)
	x = np.dot(invA,B)
	g = x[0:n+1]
	return g[:,0]

# combine pixels to reduce noise and obtain a more reliable estimation
def radiancemap_fn(images, b, rescurve, w_fn):
	shape = images.shape
	bb = np.zeros(shape,dtype = np.float32)
	for i in range(shape[1]):
		for j in range(shape[2]):
			bb[:,i,j] = b
	r_map = np.zeros(shape[1:], dtype=np.float32)
	print ("r_map shape = ", r_map.shape)
	imnum = len(images)
	g = rescurve[images]
	w = w_fn(images)
	wsum = np.sum(w,axis=0)
	r_map = np.sum(w * (g - bb) / wsum,axis = 0)
	for i in range(shape[1]):
		for j in range(shape[2]):
			if np.isnan(r_map[i][j]):
				r_map[i][j] = g[imnum//2][i][j] - bb[imnum //2][i][j]
	return np.exp(r_map)

def pixel_intensity(images):
	zmin = 0
	zmax = 255
	layer_num = len(images)
	sample_num = 256 #zmax-zmin+1
	# pick the middle one to get which pixel now
	middle = layer_num // 2
	print("images shape =" , images.shape)
	mid_img = images[middle,:,:]
	print("mid_img shape =" , mid_img.shape)
	intensity = np.zeros((sample_num, layer_num), dtype=np.uint8)
	for i in range(zmax+1):
		rows, columns = np.where(mid_img == i)
		rows_len = len(rows)
		if rows_len != 0:
			index = random.randrange(rows_len)
			r = rows[index]
			c = columns[index]
			intensity[i,:] = images[:,r, c]
	return intensity

def HDR(images, b, l):
	tmp = images[...,].shape
	print ("hdrtmp = ",tmp[1:] )
	im_hdr = np.zeros(tmp[1:],dtype=np.float32)
	for i in range(tmp[3]):
		img = images[:,:,:,i]
		Z = pixel_intensity(img)
		print ("Z shape = ", Z.shape)
		rescurve = response_fn(Z,b, l, weight_fn)
		response_list.append(rescurve) 
		radiancemap = radiancemap_fn(img, b, rescurve, weight_fn)
		print (radiancemap)
		# print (np.min(radiancemap),np.max(radiancemap))
		radiance_list.append(radiancemap)
		im_hdr[..., i] = cv2.normalize(radiancemap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
	return im_hdr

def readImagesAndTimes():
	# List of exposure times
	# times = np.log(np.array([1/160, 1/125, 1/80, 1/60, 1/40, 1/15], dtype=np.float32))
	# times = np.log(np.array([2, 1, 0.5, 0.25, 0.125, 0.062, 0.031, 0.015, 0.008, 0.004], dtype=np.float32))
	# times = np.log(np.array([1 ,2 ,4, 0.5, 0.25, 1/6, 1/13, 1/20, 1/30, 1/50, 1/80, 1/125, 1/250, 1/500, 1/1000], dtype=np.float32))
	# times = np.log(np.array([32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.15625, 0.0078125, 0.00390625, 0.001953125, 0.0009765265], dtype=np.float32))
	times = np.log(np.array([9, 4, 2.5, 1.3, 1/1.6, 1/3, 1/6, 1/13, 1/25, 1/50, 1/80, 1/125, 1/320], dtype=np.float32))
	# times = np.log(np.array([25, 15, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.005], dtype=np.float32))
	# times = np.log(np.array([6, 4, 2.5, 1, 0.5, 0.25, 1/6, 1/13, 1/20, 1/30, 1/50, 1/80, 1/125,1/250, 1/500,  1/1000], dtype=np.float32))
	# List of image filenames
	# times = np.log(np.array([1, 1/1.6, 1/2.5, 1/4, 1/6, 1/10, 1/15, 1/30, 1/50, 1/80, 1/125, 1/200, 1/320, 1/500, 1/800, 1/1600,1/3200,1/4000], dtype=np.float32))
	# times = np.log(np.array([1/1.6, 1/3, 1/5, 1/10, 1/15, 1/25, 1/50, 1/100, 1/200, 1/400, 1/800], dtype=np.float32))
	# filenames = [("./road/DSC_"+str(i)+".JPG") for i in range(3767, 3777)]
	filenames = [("./sc/DSC_"+str(i)+".JPG") for i in range(3750, 3763)]	
	images = []

	for filename in filenames:
		im = cv2.imread(filename)
		images.append(im)
	images = np.array(images)
	return images, times

img_list, exp_times = readImagesAndTimes()

response_list = []
radiance_list = []
result_img = HDR(img_list, exp_times, float(150))
plt.imsave('./hdr_sc.png',result_img)
# tonemapping:
# Tonemap using Drago's method to obtain 24-bit color image
tonemapDrago = cv2.createTonemapDrago(0.787, 0.95)
ldrDrago = tonemapDrago.process(result_img)
ldrDrago = 3 * ldrDrago
cv2.imwrite("./Drago_sc.png", ldrDrago*255)
# Tonemap using Reinhard's method to obtain 24-bit color image
'''
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(result_img)
cv2.imwrite("./Reinhard_sc.jpg", ldrReinhard * 255)
'''