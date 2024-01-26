from math import sqrt  # atan

import cv2
import numpy as np

from tools.filters import bright_contrast, histogram_adp
from tools.processtool import get_dist_to_center, mask_by_color, mask_center

# сonstants
planewidth = 1200
planeheight = 800
dist_to_plane = 2200
min_accur = 50

cross2squareK = 20 / 15  # 20 / 15

coordinates = {"aim_center": []}
hyps = []  # list to save hypotheses to the center
sizes = {}


image = cv2.imread("rect4.png")  # photo4.jpg rect4.png
height, width = image.shape[:2]


img = histogram_adp(image)  # adaptive histogram equalization
img = bright_contrast(img, -10, 30)  # brightness and contrast


# color masks
mask_red = mask_by_color(
    img, [(0, 50, 50), (50, 255, 255)], [(140, 50, 50), (200, 255, 255)]
)
mask_blue = mask_by_color(img, [(85, 210, 30), (185, 255, 255)])
mask_white = mask_by_color(img, [(0, 3, 120), (255, 15, 255)])


center_mask = mask_center(mask_red, divk=2)  # create center mask
center_mask = cv2.cvtColor(center_mask, cv2.COLOR_BGR2GRAY)  # convert mask to grayscale

img_center_only = 255 - cv2.bitwise_and(mask_red, center_mask)  # apply center mask


def myround(x, base=5):
    return base * round(x / base)


# Init cascade
cascade = cv2.CascadeClassifier()
cascade.load("cascade.xml")  # load cascade

# Detect objects using cascade
detected_rects = cascade.detectMultiScale(
    img_center_only,
    scaleFactor=1.8,
    minNeighbors=5,
    minSize=(50, 50),
    maxSize=(150, 150),
    flags=cv2.CASCADE_SCALE_IMAGE,
)

print(f"Found {len(detected_rects)} objects using haar cascade!")


# Get the nearest to the real center rect and save it to 'haar_rect'
haar_rect = {"i": 0, "val": 100000, "center": (0, 0), "wh": (0, 0)}
for i, (dX, dY, dW, dH) in enumerate(detected_rects):
    # draw the face bounding box on the frame
    cv2.rectangle(img, (dX, dY), (dX + dW, dY + dH), (30, 200, 230), 2)

    plus_center = (dX + dW // 2, dY + dH // 2)

    # check if new center is closer to the real center than previous one
    if haar_rect["val"] > sqrt(
        (dX + dW / 2 - width / 2) ** 2 + (dY + dH / 2 - height / 2) ** 2
    ):
        haar_rect["val"] = sqrt(
            (dX + dW / 2 - width / 2) ** 2 + (dY + dH / 2 - height / 2) ** 2
        )
        haar_rect["center"] = plus_center
        haar_rect["wh"] = (dW, dH)
        haar_rect["i"] = i


contours, _ = cv2.findContours(
    255 - img_center_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
for cnt in contours:
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])  # get center X
    cY = int(M["m01"] / M["m00"])  # get center Y
    cnt_center = [cX, cY]

    # generate range of X and Y coordinates of the haar contour
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

sizes["square"] = (
    int(sizes["plus"][0] * cross2squareK),
    int(sizes["plus"][1] * cross2squareK),
)

print("Width of the square: ", sizes["square"][0])
print("Height of the square: ", sizes["square"][1])

sizes["plane"] = (sizes["square"][0] * 6, sizes["square"][1] * 4)

print("Width of the plane: ", sizes["plane"][0])
print("Height of the plane: ", sizes["plane"][1])

# crop img by plane sizes
img = img[
    coordinates["center"][1]
    - sizes["plane"][1] // 2 : coordinates["center"][1]
    + sizes["plane"][1] // 2,
    coordinates["center"][0]
    - sizes["plane"][0] // 2 : coordinates["center"][0]
    + sizes["plane"][0] // 2,
]

mask_blue = mask_blue[
    coordinates["center"][1]
    - sizes["plane"][1] // 2 : coordinates["center"][1]
    + sizes["plane"][1] // 2,
    coordinates["center"][0]
    - sizes["plane"][0] // 2 : coordinates["center"][0]
    + sizes["plane"][0] // 2,
]

# recalculate center coordinates
coordinates["center"][0] = sizes["plane"][0] / 2
coordinates["center"][1] = sizes["plane"][1] / 2

kernel = np.ones((3, 3), np.uint8)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2)
mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=4)

cv2.imwrite("output2.jpg", mask_blue)


rows = mask_blue.shape[0]
circles = cv2.HoughCircles(
    mask_blue,
    cv2.HOUGH_GRADIENT,
    1,
    rows / 8,
    param1=1000,
    param2=6,
    minRadius=round(sizes["square"][0] / 5),
    maxRadius=sizes["square"][0],
)

# print(circles)

res = np.zeros(img.shape)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        center = (i[0], i[1])
        coordinates["aim_center"].append([i[0], i[1]])
        cv2.circle(img, center, 4, (0, 255, 0), 6)
        radius = i[2]
        cv2.circle(img, center, radius, (255, 0, 255), 3)


for i, coord in enumerate(coordinates["aim_center"]):
    h = get_dist_to_center(
        coord[0] - coordinates["center"][0], coord[1] - coordinates["center"][1]
    )
    h = myround(h, 25)
    hyps.append([i, h])

hyps.sort(key=lambda x: x[1])  # sort by hyp value

# find same values
for i in range(len(hyps) - 1):
    if hyps[i][1] == hyps[i + 1][1]:
        # indexes
        a = hyps[i][0]
        b = hyps[i + 1][0]

        # coordinates['aim_center'][a][0] or coordinates['aim_center'][b][0] is greater
        if coordinates["aim_center"][a][0] > coordinates["aim_center"][b][0]:
            a, b = b, a
            hyps[i], hyps[i + 1] = hyps[i + 1], hyps[i]

        print(
            "X coord first:",
            coordinates["aim_center"][a][0],
            "   X coord second:",
            coordinates["aim_center"][b][0],
        )

print(hyps)
# put text num
for i in range(len(hyps)):
    cv2.putText(
        img,
        str(i),
        (
            coordinates["aim_center"][hyps[i][0]][0],
            coordinates["aim_center"][hyps[i][0]][1] - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
cv2.imwrite("output2.jpg", img)
