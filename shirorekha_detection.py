import cv2

uppers = []

def regenerating_image(address, filename):
    print("Detecting lines from " + filename)
    destination = "".join([address, filename])

    # Reading the image
    img = cv2.imread(destination)

    # Convert to Gray Scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur for noise reduction (5,5) 0 => default
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image using a fixed threshold (you can adjust the threshold value)
    _, threshed = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # OTSU threshold seems best threshold for line detection
    th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find and draw the upper boundary of each line
    hist = cv2.reduce(threshed, 1, cv2.REDUCE_AVG).reshape(-1)
    th_H = 100
    th_L = 25
    H, W = img.shape[:2]
    uppers = [y for y in range(H - 1) if hist[y] < th_H < hist[y + 1]]

    # Create a copy of the original image to draw lines on
    img_with_lines = img.copy()

    y=uppers[0]
    cv2.line(img_with_lines, (0, y), (W, y), (255, 255, 255), 70)
    y=uppers[-1]
    cv2.line(img_with_lines, (0, y), (W, y), (255, 255, 255), 70)

        # Save image with lines
    cv2.imwrite(address + "result.png", img_with_lines)
    
    return uppers


# Usage
address = "C:/Users/User/Desktop/"
filename = "vrinda_2.jpeg"
uppers =regenerating_image(address, filename)
print(uppers)


################################################################################

# Set image path
fileName = "C:/Users/User/Desktop/result.png"
#fileName = "C:/Users/User/Desktop/x.jpg"
# Read input image
inputImage = cv2.imread(fileName)
inputCopy = inputImage.copy()

# Convert BGR to grayscale
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Set the adaptive thresholding (Gaussian) parameters
windowSize = 31
windowConstant = -1

# Apply the threshold
binaryImage = cv2.adaptiveThreshold(
    grayscaleImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, windowSize, windowConstant
)

# Set kernel (structuring element) size
kernelSize = 3

# Set operation iterations
opIterations = 2

# Get the structuring element
maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))

# Perform closing
closingImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, maxKernel, None, None, opIterations,
                                cv2.BORDER_REFLECT101)

# Find contours of characters
contours, hierarchy = cv2.findContours(closingImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define upper and lower lines (you need to adjust these values)
# upper_line = 250  # Y-coordinate of upper line
# lower_line = 500  # Y-coordinate of lower line

# Initialize lists to store upper, lower, and middle characters
upper_chars = []
lower_chars = []
middle_chars = []

# Iterate through contours and classify characters
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    center_y = y + h // 2

    # Classify characters based on their position relative to the lines
    if center_y < uppers[0]:
        upper_chars.append((x, y, w, h))
    elif center_y > uppers[-1]:
        lower_chars.append((x, y, w, h))
    else:
        middle_chars.append((x, y, w, h))

# Draw bounding boxes for upper characters
for x, y, w, h in upper_chars:
    color = (0, 0, 255)  # Red
    cv2.rectangle(inputCopy, (x, y), (x + w, y + h), color, 2)

# Draw bounding boxes for lower characters
for x, y, w, h in lower_chars:
    color = (0, 255, 0)  # Green
    cv2.rectangle(inputCopy, (x, y), (x + w, y + h), color, 2)

# Draw bounding boxes for middle characters
for x, y, w, h in middle_chars:
    color = (255, 0, 0)  # Blue
    cv2.rectangle(inputCopy, (x, y), (x + w, y + h), color, 2)

for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    center_y = y + h // 2

    # Classify characters based on their position relative to the lines
    if center_y < uppers[0]:
        zone = "upper"
    elif center_y > uppers[-1]:
        zone = "lower"
    else:
        zone = "middle"

    # Generate file name based on zone and position
    file_name = "{}_{}_{}.png".format(zone, x, x + w)

    # Crop the character and save as file
    char_image = inputImage[y:y + h, x:x + w]
    cv2.imwrite(file_name, char_image)

# Save or display the result
cv2.imwrite("output.png", inputCopy)
cv2.imshow("Result", inputCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

