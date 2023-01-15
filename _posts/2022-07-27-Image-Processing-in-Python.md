---
layout: xxxx
title: Image Processing in Python
date: 2022-07-27
categories: image, python, scikit-image
comments: true
---
## Intro
* Scikit-image
* 
![](https://i.imgur.com/4f5odm3.png)

* 2-D matrix
	* RGB channels
		* 3 channels

![](https://i.imgur.com/usNDp8X.png)


![](https://i.imgur.com/4zGErD1.png)


### Numpy for images
```
matplotlib.imread([image])
```

![](https://i.imgur.com/RKUNXJP.png)


![](https://i.imgur.com/PQViFIE.png)


```python
vertically_flipped = np.flipud(madrid_image)

horizontally_flipped = np.fliplr(madrid_image)

```

![](https://i.imgur.com/HnaSGfv.png)



![](https://i.imgur.com/QazQ4Jg.png)

![](https://i.imgur.com/rsFKu7K.png)


### Thresholding
![](https://i.imgur.com/DzTDvfh.png)


![](https://i.imgur.com/PGm7pdT.png)


	* Inverted thresholding
![](https://i.imgur.com/oeYsGAg.png)


* thresholding algorithm
	* global
![](https://i.imgur.com/Vei2CQH.png)


![](https://i.imgur.com/b7XhVsP.png)

    
    * local
![](https://i.imgur.com/7lt6rtX.png)


![](https://i.imgur.com/V3CEDoR.png)


* optimal thresh
```python

# Import threshold and gray convertor functions
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Turn the image grayscale
gray_tools_image = rgb2gray(tools_image)

# Obtain the optimal thresh
thresh = threshold_otsu(gray_tools_image)

# Obtain the binary image by applying thresholding
binary_image = gray_tools_image > thresh

# Show the resulting binary image
show_image(binary_image, 'Binarized image')


```

![](https://i.imgur.com/WE56BXi.png)

![](https://i.imgur.com/Iv6MgZG.png)


## *Filters, Contrast, Transformation and Morphology*
### Filtering
	* Enhancing an image
	* Emphasize or remove features
	* Smoothing
	* Sharpening
	* Edge detection

* Sobel : edge detection

![](https://i.imgur.com/gYbFPhd.png)


![](https://i.imgur.com/F2upzl8.png)


![](https://i.imgur.com/eKvKck7.png)


```python

# Import the color module
from skimage import color

# Import the filters module and sobel function
from skimage.filters import sobel

# Make the image grayscale
soaps_image_gray = color.rgb2gray(soaps_image)

# Apply edge detection filter
edge_sobel = sobel(soaps_image_gray)

# Show original and resulting image to compare
show_image(soaps_image, "Original")
show_image(edge_sobel, "Edges with Sobel")

```
![](https://i.imgur.com/6ouzMfc.png)

![](https://i.imgur.com/efI5nxx.png)



* Gaussian smoothing
![](https://i.imgur.com/FZWAqYe.png)


![](https://i.imgur.com/Swn0iMw.png)


```python
# Import Gaussian filter 
from skimage.filters import gaussian

# Apply filter
gaussian_image = gaussian(building_image, multichannel=True)

# Show original and resulting image to compare
show_image(building_image, "Original")
show_image(gaussian_image, "Reduced sharpness Gaussian")


```

![](https://i.imgur.com/WQvgX2o.png)

![](https://i.imgur.com/AIBasoc.png)


### Contrast Enhancement
![](https://i.imgur.com/faeTTHt.png)

```python
# Import the required module
from skimage import exposure

# Show original x-ray image and its histogram
show_image(chest_xray_image, 'Original x-ray')

plt.title('Histogram of image')
plt.hist(chest_xray_image.ravel(), bins=256)
plt.show()
```
![](https://i.imgur.com/HYhNhP1.png)


![](https://i.imgur.com/XPyDvVA.png)

![](https://i.imgur.com/04Y3hjO.png)


![](https://i.imgur.com/OuhVE0E.png)


* Example
```python 
# Import the necessary modules
from skimage import data, exposure

# Load the image
original_image = data.coffee()

# Apply the adaptive equalization on the original image
adapthist_eq_image = exposure.equalize_adapthist(original_image, clip_limit=0.03)

# Compare the original image to the equalized
show_image(original_image)
show_image(adapthist_eq_image, '#ImageProcessingDatacamp')


```
![](https://i.imgur.com/Qhz3OfC.png)

![](https://i.imgur.com/edjVf7r.png)


### Transformation
![](https://i.imgur.com/TnTl7lu.png)


* Rotating clockwise
![](https://i.imgur.com/HIeejgj.png)

* Rescaling : downgrading
![](https://i.imgur.com/51FQMn1.png)

![](https://i.imgur.com/0Vwibi3.png)

* Resize
![](https://i.imgur.com/6soTvrD.png)

![](https://i.imgur.com/Rc3Kyx6.jpg)



* example
```python
# Import the module and the rotate and rescale functions
from skimage.transform import rotate, rescale

# Rotate the image 90 degrees clockwise 
rotated_cat_image = rotate(image_cat, -90)

# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=True, multichannel=True)

# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, anti_aliasing=False, multichannel=True)

# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")
```

![](https://i.imgur.com/gq5enHd.png)

![](https://i.imgur.com/2md5FSX.png)


```python
# Import the module and function to enlarge images
from skimage.transform import rescale

# Import the data module
from skimage import data

# Load the image from data
rocket_image = data.rocket()

# Enlarge the image so it is 3 times bigger
enlarged_rocket_image = rescale(rocket_image, 3, anti_aliasing=True, multichannel=True)

# Show original and resulting image
show_image(rocket_image)
show_image(enlarged_rocket_image, "3 times enlarged image")


```

![](https://i.imgur.com/jB69Jey.png)

![](https://i.imgur.com/0SLS3PZ.png)


### Morphology

![](https://i.imgur.com/iROgxyG.png)


* Erosion
![](https://i.imgur.com/q8UFeSN.png)


* Dilation
* 
![](https://i.imgur.com/CBDNDpf.png)

![](https://i.imgur.com/MFSIoR5.png)


## Image restoration, Noise, Segmentation and Contours*
### Image Restoration
```python
# Import the module from restoration
from skimage.restoration import inpaint

# Show the defective image
show_image(defect_image, 'Image to restore')

# Apply the restoration function to the image using the mask
restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)
show_image(restored_image)

```

![](https://i.imgur.com/SOwPAuC.png)

![](https://i.imgur.com/DRSmEpq.png)


* Remove logos
```python
# Initialize the mask
mask = np.zeros(image_with_logo.shape[:-1])

# Set the pixels where the logo is to 1
mask[210:290, 360:425] = 1

# Apply inpainting to remove the logo
image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo, 
                                                mask, 
                                                multichannel=True)

# Show the original and logo removed images
show_image(image_with_logo, ‘Image with logo’)
show_image(image_logo_removed, ‘Image with logo removed’)

```

![](https://i.imgur.com/DLCmfL7.png)



### Noise
![](https://i.imgur.com/2hglvsm.png)



* Denoising types
	* Total variation
	* Bilateral
	* wavelet denoising
	* Non-local means denoising

![](https://i.imgur.com/X0xMvXS.png)


* Example
```python
# Import the module and function
from skimage.util import random_noise

# Add noise to the image
noisy_image = random_noise(fruit_image)

# Show original and resulting image
show_image(fruit_image, 'Original')
show_image(noisy_image, 'Noisy image')

```

```python
# Import the module and function
from skimage.restoration import denoise_tv_chambolle

# Apply total variation filter denoising
denoised_image = denoise_tv_chambolle(noisy_image, 
                                      multichannel=True)

# Show the noisy and denoised images
show_image(noisy_image, 'Noisy')
show_image(denoised_image, 'Denoised image')

```

![](https://i.imgur.com/zMhowMk.png)

![](https://i.imgur.com/0iIArrt.png)


```python
# Import bilateral denoising function
from skimage.restoration import denoise_bilateral

# Apply bilateral filter denoising
denoised_image = denoise_bilateral(landscape_image, 
                                   multichannel=True)

# Show original and resulting images
show_image(landscape_image, 'Noisy image')
show_image(denoised_image, 'Denoised image')

```


![](https://i.imgur.com/laTlGcl.png)

![](https://i.imgur.com/mNikrMz.png)


### *Superpixels & segmentation*
* Segmentation
	* Thresholding
![](https://i.imgur.com/UGZ5Y1t.png)


	* superpixels
![](https://i.imgur.com/OprP7pt.png)


* Types
	* Supervised
	* Unsupervised
![](https://i.imgur.com/vHuqG59.png)

		* use “K-means”
![](https://i.imgur.com/4QS1uPO.png)

![](https://i.imgur.com/xSIUuVt.png)

![](https://i.imgur.com/AHZFq0R.png)


* Example : *Superpixel segmentation*
```python
# Import the slic function from segmentation module
from skimage.segmentation import slic

# Import the label2rgb function from color module
from skimage.color import label2rgb

# Obtain the segmentation with 400 regions
segments = slic(face_image, n_segments= 400)

# Put segments on top of original image to compare
segmented_image = label2rgb(segments, face_image, kind=‘avg’)

# Show the segmented image
show_image(segmented_image, “Segmented image, 400 superpixels”)

```

![](https://i.imgur.com/WRsQa3J.png)

![](https://i.imgur.com/WMJ3EQz.png)


### Finding contours 輪廓
* Binary images
	* thresholding
	* Edge detection
* Preprocessing
	* Transform to 2D grayscale 
![](https://i.imgur.com/DrU71vk.png)


	* Binarize the image
![](https://i.imgur.com/2nfMo5V.png)

	* Find contours
![](https://i.imgur.com/ln0e1C1.png)


* Contour’s shape
![](https://i.imgur.com/i9mHezY.png)


* count the dots in a dice’s image
```python
# Create list with the shape of each contour 
shape_contours = [cnt.shape[0] for cnt in contours]

# Set 50 as the maximum size of the dots shape
max_dots_shape = 50

# Count dots in contours excluding bigger than dots size
dots_contours = [cnt for cnt in contours if np.shape(cnt)[0] < max_dots_shape]

# Shows all contours found 
show_image_contour(binary, contours)

# Print the dice's number
print("Dice's dots number: {}. ".format(len(dots_contours)))

    Dice's dots number: 9. 

```

## *Advanced Operations, Detecting Faces and Features*
* Edge detection 
	* sobel
	* Canny
![](https://i.imgur.com/x8Falf9.png)

![](https://i.imgur.com/xrOhFrd.png)


![](https://i.imgur.com/l0l6KVa.png)

![](https://i.imgur.com/BV42LFd.png)


* Example 
```python
# Import the canny edge detector 
from skimage.feature import canny

# Convert image to grayscale
grapefruit = color.rgb2gray(grapefruit)

# Apply canny edge detector
canny_edges = canny(grapefruit)

# Show resulting image
show_image(canny_edges, "Edges with Canny")


# Apply canny edge detector with a sigma of 1.8
edges_1_8 = canny(grapefruit, sigma=1.8)

# Apply canny edge detector with a sigma of 2.2
edges_2_2 = canny(grapefruit, sigma=2.2)

# Show resulting images
show_image(edges_1_8, "Sigma of 1.8")
show_image(edges_2_2, "Sigma of 2.2")

```

* Corner detetion 
	* motion detetion
	* image alignment
	* video tracking
	* 3d modeling
	* object detection

* Matching corners

![](https://i.imgur.com/cZiG6pD.png)

![](https://i.imgur.com/HHh8ua6.png)


* Harris corner detector
![](https://i.imgur.com/1kPaS05.png)

![](https://i.imgur.com/qR8oCXD.png)


![](https://i.imgur.com/3RHSSYB.png)

![](https://i.imgur.com/MOA7niU.png)


![](https://i.imgur.com/cgelbeZ.png)


* example
```python
# Import the corner detector related functions and module
from skimage.feature import corner_harris, corner_peaks

# Convert image from RGB-3 to grayscale
building_image_gray = color.rgb2gray(building_image)

# Apply the detector  to measure the possible corners
measure_image = corner_harris(building_image_gray)

# Find the peaks of the corners
coords = corner_peaks(measure_image, min_distance=20, threshold_rel=0.02)

# Show original and resulting image with corners detected
show_image(building_image, "Original")
show_image_with_corners(building_image, coords)


```


### Face detection
![](https://i.imgur.com/4GSW6cg.png)


![](https://i.imgur.com/ikQ4f4A.png)


![](https://i.imgur.com/eHxvgyo.png)


* Example
```python
# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with min and max size of searching window
detected = detector.detect_multi_scale(img = night_image,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(10, 10),
                                       max_size=(200, 200))

# Show the detected faces
show_detected_face(night_image, detected)

```


![](https://i.imgur.com/WX3cr2g.png)

![](https://i.imgur.com/fB76uZO.png)

![](https://i.imgur.com/HDsw6uq.png)



```python
# Load the trained file from data
trained_file = data.lbp_frontal_face_cascade_filename()

# Initialize the detector cascade
detector = Cascade(trained_file)

# Detect faces with scale factor to 1.2 and step ratio to 1
detected = detector.detect_multi_scale(img=friends_image,
                                       scale_factor=1.2,
                                       step_ratio=1,
                                       min_size=(10, 10),
                                       max_size=(200, 200))
# Show the detected faces
show_detected_face(friends_image, detected)

<script.py> output:
    {'r': 218, 'c': 440, 'width': 52, 'height': 52}
    {'r': 202, 'c': 402, 'width': 45, 'height': 45}
    {'r': 207, 'c': 152, 'width': 47, 'height': 47}
    {'r': 217, 'c': 311, 'width': 39, 'height': 39}
    {'r': 219, 'c': 533, 'width': 48, 'height': 48}
    {'r': 202, 'c': 31, 'width': 36, 'height': 36}
    {'r': 242, 'c': 237, 'width': 41, 'height': 41}
```


![](https://i.imgur.com/sUvzGAm.png)


* segmentation and face detection
```python
# Obtain the segmentation with default 100 regions
segments = slic(profile_image)

# Obtain segmented image using label2rgb
segmented_image = label2rgb(segments, profile_image, kind='avg')

# Detect the faces with multi scale method
detected = detector.detect_multi_scale(img=segmented_image, 
                                       scale_factor=1.2, 
                                       step_ratio=1, 
                                       min_size=(10, 10), max_size=(1000, 1000))

# Show the detected faces
show_detected_face(segmented_image, detected)

```

![](https://i.imgur.com/hALF0hn.png)

![](https://i.imgur.com/X9gP6X4.png)


### Real-world applications
![](https://i.imgur.com/5ohI0aY.png)


* Privacy protection
```python
# Detect the faces
detected = detector.detect_multi_scale(img=group_image, 
                                       scale_factor=1.2, step_ratio=1, 
                                       min_size=(10, 10), max_size=(100, 100))
# For each detected face
for d in detected:  
    # Obtain the face rectangle from detected coordinates
    face = getFaceRectangle(d)
    
    # Apply gaussian filter to extracted face
    blurred_face = gaussian(face, multichannel=True, sigma = 8)
    
    # Merge this blurry face to our final image and show it
    resulting_image = mergeBlurryFace(group_image, blurred_face) 
show_image(resulting_image, "Blurred faces")

```

![](https://i.imgur.com/LgdsbOX.png)

![](https://i.imgur.com/TKSoWTI.png)


* *Help Sally restore her graduation photo*

```python
# Import the necessary modules
from skimage.restoration import denoise_tv_chambolle, inpaint
from skimage.transform import rotate

# Transform the image so it's not rotated
upright_img = rotate(damaged_image, 20)

# Remove noise from the image, using the chambolle method
upright_img_without_noise = denoise_tv_chambolle(upright_img,weight=0.1, multichannel=True)

# Reconstruct the image missing parts
mask = get_mask(upright_img)
result = inpaint.inpaint_biharmonic(upright_img_without_noise, mask, multichannel=True)

show_image(result)

```

![](https://i.imgur.com/hp2JDJB.png)

![](https://i.imgur.com/HDIkNCX.png)


