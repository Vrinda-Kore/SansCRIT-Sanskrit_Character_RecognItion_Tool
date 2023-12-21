import cv2
import numpy as np
import os
from PIL import Image

# work on individual image
'''
img = cv2.imread('100a_1.jpg')
img1 = Image.open('100a_1.jpg')
fn = img1.filename

grid_size = 11
# print(img.shape[2])
# Split the image into smaller sub-images
sub_images = [img[x:x + img.shape[0] // grid_size, y:y + img.shape[1] // grid_size] for x in
              range(0, img.shape[0], img.shape[0] // grid_size) for y in
              range(0, img.shape[1], img.shape[1] // grid_size)]
# Save the sub-images
for i, sub_image in enumerate(sub_images):
    cv2.imwrite(f"{fn[:-4]}_{i + 1}.png", sub_image)



# final: working on a folder containing both 11x11 and 3x9 images and storing the split images in a pre-existing 'cropped' folder

import os
import cv2

folder_path = ''
output_folder = 'cropped'
#grid_size = 11
# Loop through all the files in the folder
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Open the image using OpenCV
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        
        basename, ext = os.path.splitext(filename)
        if basename[3] == 'a':
            sub_images1 = [img[x:x + img.shape[0] // 11, y:y + img.shape[1] // 11] for x in
                           range(0, img.shape[0], img.shape[0] // 11) for y in
                           range(0, img.shape[1], img.shape[1] // 11)]
            for j, sub_image in enumerate(sub_images1):
                output_filename = f"{basename}_{j + 1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, sub_image)
        if basename[3] == 'b':
            sub_images2 = [img[x:x + img.shape[0] // 3, y:y + img.shape[1] // 9] for x in
                           range(0, img.shape[0], img.shape[0] // 3) for y in range(0, img.shape[1], img.shape[1] // 9)]
            for j, sub_image in enumerate(sub_images2):
                output_filename = f"{basename}_{j + 1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, sub_image)


# grayscaling single image

img1 = Image.open('r_125.png')
gray_img = img1.convert('L')
gray_img.save('r_125_gray.png')

# work on grayscaling all the images present in a folder

input_folder = 'cropped'
output_folder = 'grayscaled'

# Get a list of all image files in the input folder
files = os.listdir(input_folder)
image_files = [file for file in files if file.endswith(('jpg', 'jpeg', 'png'))]

# Looping through each image file and grayscaling it
for file in image_files:
    img_path = os.path.join(input_folder, file)
    img = Image.open(img_path)
    gray_img = img.convert('L')
    output_path = os.path.join(output_folder, file)
    gray_img.save(output_path)
    

# resizing the image + grayscaling the image

# Set the input and output folder paths
input_folder = '16428_images_zipped'
output_folder = 'grayscaled_resized'

# Get a list of all image files in the input folder
files = os.listdir(input_folder)
image_files = [file for file in files if file.endswith(('jpg', 'jpeg', 'png'))]

for file in image_files:
    # Load the image

    img_path = os.path.join(input_folder, file)
    img = Image.open(img_path)
    #img = cv2.imread(img_path, 0)
    # Resize the image to fit within a 64x64 pixel box while preserving aspect ratio
    #resized_img = cv2.resize(img, (64, 64))
    
    # Convert the image to grayscale
    gray_img = img.convert('L')
    
    resized = img.resize((64,64))
    resized.save(img_path)
    # Save the grayscale image
    #output_path = os.path.join(output_folder, file)
    #gray_img.save(img_path)

'''
'''
#cropping the image further to remove the borders

# Importing Image class from PIL module
from PIL import Image

# Opens a image in RGB mode
im = Image.open('001a_1_3.jpg')
width, height = im.size
# Setting the points for cropped image
left = 5
top = height / 6
right = 180
bottom = height

# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))

# Shows the image in image viewer
im1.show()



input_folder = 'random'
output_folder = 'random2'

# Get a list of all image files in the input folder
files = os.listdir(input_folder)
image_files = [file for file in files if file.endswith(('jpg', 'jpeg', 'png'))]

for file in image_files:
    # Load the image
    img_path = os.path.join(input_folder, file)
    img = Image.open(img_path)

    width, height = img.size
    # Setting the points for cropped image
    left = 20
    top = height / 20
    right = 100
    bottom = height 

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = img.crop((left, top, right, bottom))

    # Save the grayscale image
    output_path = os.path.join(output_folder, file)
    im1.save(output_path)
'''

'''
import cv2
import numpy as np

img = cv2.imread('001a_1_85.jpg', 0)
_, blackAndWhite = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
sizes = stats[1:, -1]  # get CC_STAT_AREA component
img2 = np.zeros(labels.shape, np.uint8)

for i in range(0, nlabels - 1):
    if sizes[i] >= 50:  # filter small dotted regions
        img2[labels == i + 1] = 255

res = cv2.bitwise_not(img2)

cv2.imwrite('res.png', res)
'''

