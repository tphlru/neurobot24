from math import sqrt  # atan

import cv2
import numpy as np

from tools.filters import bright_contrast, histogram_adp
from tools.processtool import get_dist_to_center, mask_by_color, mask_center, invert
from os import environ

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

# сonstants
planewidth = 260
planeheight = 170
dist_to_plane = 2200  # стандартное?
min_accur = 50
cal_k = float(input("Enter CAL_K value: ") or "1500" if not (environ.get('CAL_K')) else environ.get('CAL_K'))  # 1.0
pix_k = 1
size_plus = 28

cross2squareK: float = 43.5 / 32  # 20 / 15

color_ranges = {
    "white": [(0, 0, 90), (255, 153, 255)],
    "red": [[(0, 50, 50), (50, 255, 255)], [(140, 50, 50), (200, 255, 255)]],
    "blue": [(85, 102, 40), (194, 255, 255)],
    "black": [(0, 0, 0), (255, 255, 36)],
}

coordinates = {"aim_center": []}
hyps: list = []  # list to save hypotheses to the center
sizes = {}


def show(simg):
    cv2.imshow("image", cv2.resize(simg, (width // 2, height // 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# read frame from cam
cam = cv2.VideoCapture(0)
# set cam params
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cam.read()
cam.release()

# image = cv2.imread("rect5.png")  # photo4.jpg rect4.png
image = frame
height, width = image.shape[:2]
show(image)

# FILTERS
img = histogram_adp(image)  # adaptive histogram equalization
# show(img)

img = bright_contrast(img, 0, threshold_yen(img) // 6)  # brightness and contrast
# print(yen_threshold)
show(img)

# IMAGE PROCESSING
mask_white = invert(mask_by_color(img, color_ranges['white']))  # Shades of white (and invert)
img[mask_white == 0] = (255, 255, 255)  # (apply mask) Now true white

mask_red = mask_by_color(img, color_ranges['red'])  # Shades of red and pink
mask_blue = mask_by_color(img, color_ranges['blue'])  # Shades of blue
mask_black = (mask_by_color(img, color_ranges['black']))  # Shades of black

center_mask = mask_center(mask_red, divk=2)  # Create center mask (half of wd and half of ht)
center_mask = cv2.cvtColor(center_mask, cv2.COLOR_BGR2GRAY)  # Convert center mask to grayscale

img_center_only = invert(cv2.bitwise_and(mask_red, center_mask))  # Apply center mask and invert

kernel = np.ones((3, 3), np.uint8)  # Generate "drawing" kernel

# This will make the image of the red cross clearer and more integral
img_center_only = invert(cv2.morphologyEx(invert(img_center_only), cv2.MORPH_DILATE, kernel, iterations=4))

# This will remove minor noise
# mask_black = (cv2.morphologyEx(invert(mask_black), cv2.MORPH_BLACKHAT, kernel, iterations=10))

# Remove all the red-colored things from blue mask
mask_blue = cv2.subtract(invert(mask_red), invert(mask_blue))
# Close possible holes in the aims
mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=3))

# Remove all the black-colored things from the blue mask
img_center_only = invert(cv2.subtract(invert(mask_black), img_center_only))
show(mask_black)
show(img_center_only)
show(mask_blue)


def myround(x, base=5):
    return base * round(x / base)


# Init cascade
# sourcery skip: remove-unnecessary-else, swap-if-else-branches
cascade = cv2.CascadeClassifier()
cascade.load("cascade.xml")  # load cascade

# Detect cross using cascade
detected_rects = cascade.detectMultiScale(
    img_center_only,
    scaleFactor=1.6,
    minNeighbors=6,
    minSize=(50, 50),
    maxSize=(220, 220),
)

print(f"Found {len(detected_rects)} objects using haar cascade!")

# Get the nearest to the real center rect and save it to 'haar_rect'
haar_rect = {"i": 0, "val": 100000, "center": (0, 0), "wh": (0, 0)}

for i, (dX, dY, dW, dH) in enumerate(detected_rects):
    # Draw the face bounding box on the frame

    # Calculate the center of the rectangle
    plus_center = (dX + dW // 2, dY + dH // 2)

    # Check if new center is closer to the real center than previous one
    distance = sqrt(plus_center[0] ** 2 + plus_center[1] ** 2)
    if distance < haar_rect["val"] or haar_rect["val"] == 0:
        # Update the haar_rect dictionary with the new closest rectangle's information
        haar_rect["val"] = distance
        haar_rect["center"] = plus_center
        haar_rect["wh"] = (dW, dH)
        haar_rect["xy"] = (dX, dY)
        haar_rect["i"] = i

cv2.rectangle(img,
              (haar_rect["xy"][0], haar_rect["xy"][1]),
              (haar_rect["xy"][0] + haar_rect["wh"][0], haar_rect["xy"][1] + haar_rect["wh"][1]),
              (30, 200, 230),
              2)

contours, _ = cv2.findContours(
    invert(img_center_only), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = list(contours)  # conv to list

print(f"Found {len(contours)} objects using contours!")
# show(cv2.drawContours(image, contours, -1, (0, 255, 0), 3))

contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
print("conts", [cv2.contourArea(c) for c in contours])

for cnt in contours:
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])  # get center X
        cY = int(M["m01"] / M["m00"])  # get center Y
    else:
        continue
    cnt_center = [cX, cY]

    # generate ranges of X and Y coordinates of the haar zone
    rX = range(
        haar_rect["center"][0] - haar_rect["wh"][0] // 2,
        haar_rect["center"][0] + haar_rect["wh"][0] // 2,
    )
    rY = range(
        haar_rect["center"][1] - haar_rect["wh"][1] // 2,
        haar_rect["center"][1] + haar_rect["wh"][1] // 2,
    )

    if cnt_center[0] in rX and cnt_center[1] in rY:
        print("Found countour in the range of the haar zone!")

        _, _, cW, cH = cv2.boundingRect(cnt)  # get width and height
        sizes["plus"] = (cW, cH)  # save width and height
        coordinates["center"] = cnt_center  # save center

        # draw the contour and center on the frame
        cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
        cv2.circle(img, cnt_center, 1, (0, 255, 0), 10)
        break

print(coordinates)

# draw
cv2.rectangle(
    img,
    (
        coordinates["center"][0] - sizes["plus"][0] // 2,
        coordinates["center"][1] - sizes["plus"][1] // 2,
    ),
    (
        (coordinates["center"][0] - sizes["plus"][0] // 2) + sizes["plus"][0],
        (coordinates["center"][1] - sizes["plus"][1] // 2) + sizes["plus"][1],
    ),
    (255, 255, 0),
    2,
)

print("Width of the plus: ", sizes["plus"][0])
print("Height of the plus: ", sizes["plus"][1])
pix_k = size_plus / ((sizes["plus"][0] + sizes["plus"][1]) / 2)
print("cm in 1 pix (pix_k): ", pix_k)

if "y" in input("Калибровать y/n"):
    dist_to_cal = float(input("Введите расстояние до калибровачной плокости в см, в формате xx.x"))
    cal_k = ((sizes["plus"][0] + sizes["plus"][1]) / 2) * dist_to_cal
    print("Калибровачный коэффицент =", cal_k)
    input("Send any to continue ...")
else:  # мы видим настоящий плюсик
    dist_to_plane = cal_k / ((sizes["plus"][0] + sizes["plus"][1]) / 2)
    print("Расстояние до рабочей плоскости =", dist_to_plane)

sizes["square"] = (
    int(sizes["plus"][0] * cross2squareK),
    int(sizes["plus"][1] * cross2squareK),
)

print("Width of the square: ", sizes["square"][0])
print("Height of the square: ", sizes["square"][1])

sizes["plane"] = (sizes["square"][0] * 6, sizes["square"][1] * 4)

print("Width of the plane: ", sizes["plane"][0])
print("Height of the plane: ", sizes["plane"][1])
show(img)

# crop img by plane sizes
center_y = coordinates["center"][1]
center_x = coordinates["center"][0]
plane_y = sizes["plane"][1]
plane_x = sizes["plane"][0]

# TODO: FIXME! Cropping doesn't work.
# print(center_y, center_x, plane_y, plane_x)
# img = (img[
#        center_y - plane_y // 2: center_y + plane_y // 2,
#        center_x - plane_x // 2: center_x + plane_x // 2,
#        ])

# show(img)
# mask_blue = (mask_blue[
#              center_y - plane_y // 2: center_y + plane_y // 2,
#              center_x - plane_x // 2: center_x + plane_x // 2,
#              ])


# recalculate center coordinates
coordinates["center"][0] = sizes["plane"][0] / 2
coordinates["center"][1] = sizes["plane"][1] / 2

# kernel = np.ones((3, 3), np.uint8)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=3)  # Remove noise
show(mask_blue)
circles = cv2.HoughCircles(
    mask_blue,
    cv2.HOUGH_GRADIENT,
    1.2,
    sizes["square"][0] // 2,
    param1=1300,
    param2=7,
    minRadius=round(sizes["square"][0] / 6),
    maxRadius=round(sizes["square"][0] / 4),
)

print("Circles: ", circles)

res = np.zeros(img.shape)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # TODO : check l to center no more than d/2
        coordinates["aim_center"].append([i[0], i[1]])
        cv2.circle(img, center, 4, (0, 255, 0), 6)
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)

for i, coord in enumerate(coordinates["aim_center"]):
    h = get_dist_to_center(
        coord[0] - coordinates["center"][0], coord[1] - coordinates["center"][1]
    )
    h = myround(h, 75)
    hyps.append([i, h])

hyps.sort(key=lambda x: x[1])  # sort by hyp value

# create a dictionary to store indices of elements with the same value
same_values = {}

# populate the dictionary
for index, value in hyps:
    if value not in same_values:
        same_values[value] = []
    same_values[value].append(index)

# looks like:  {475: [1, 2], 625: [0]}
print("sames", same_values)

# iterate over the sorted hyps list
sorted_sames = []
for sames in same_values.values():
    if len(sames) > 1:
        sorted_sames = sorted(sames, key=lambda x: coordinates["aim_center"][x][0])

sorted_hyps = [sublist[0] for sublist in hyps]
print("sorted_sames", sorted_sames)
print("sorted_hyps", sorted_hyps)

samesstr = "".join(str(x) for x in sorted_sames)
hypstr = "".join(str(x) for x in sorted_hyps)

least_index = 0  # min((hypstr.index(str(val)) for val in sorted_sames))
print("least_index", least_index)

for i in range(least_index, len(sorted_sames)):
    print("change", hypstr[i], "to", samesstr[i])
    sorted_hyps[i] = int(samesstr[i])

print("sorted_hyps", sorted_hyps)

# put text num
for i in range(len(sorted_hyps)):
    cv2.putText(
        img,
        str(i),  # text
        (
            coordinates["aim_center"][sorted_hyps[i]][0],  # X coordinate
            coordinates["aim_center"][sorted_hyps[i]][1] - 20,  # Y coordinate
        ),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,  # font (handwriting)
        1.3,  # font scale
        (0, 255, 0),  # font color
        2,  # text thickness
    )

centerd_coordinates = []
for i in range(len(sorted_hyps)):
    # centeredX = (sizes["plane"][1] / 2) - coordinates["aim_center"][sorted_hyps[i]][0]
    centeredX = coordinates["aim_center"][sorted_hyps[i]][0]
    centeredX = centeredX * pix_k

    # centeredY = (sizes["plane"][0] / 2) - coordinates["aim_center"][sorted_hyps[i]][1]
    centeredY = coordinates["aim_center"][sorted_hyps[i]][1]
    centeredY = centeredY * pix_k

    centerd_coordinates.append([centeredX, centeredY])

print(centerd_coordinates)
# cv2.imwrite("output2.jpg", img)
# imshow with halfwindow size
show(img)
