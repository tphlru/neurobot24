# -*- coding: utf-8 -*-
import statistics
from math import sqrt, atan2, atan, degrees, cos, radians, acos, copysign
from os import environ
from pprint import pprint

# Commented out IPython magic to ensure Python compatibility.
import cv2
# from skimage import io, exposure, data
import numpy as np
import time
# from google.colab import drive
# from google.colab.patches import cv2_imshow
from inputimeout import inputimeout, TimeoutOccurred
from matplotlib import pyplot as plt

from tools.filters import bright_contrast  # histogram_adp
from tools.processtool import mask_by_color, mask_center, invert
import arduino

# drive.mount('/content/drive')
# os.chdir(workpath),
# %matplotlib inline

workpath = ""

arduino = arduino.Arduino(serport="/dev/ttyUSB0", baud=9600)
arduino.init_arduino()

coordinates = {"aim_center": []}
hyps: list = []  # list to save hypotheses to the center
sizes = {}

color_ranges = {
    "white": [(0, 0, 90), (255, 130, 255)],  # inscrease 130 here for better white detection
    # "red": [[(0, 94, 68), (23, 255, 195)], [(160, 30, 68), (250, 255, 195)]],
    # "red": [(0, 49, 75), (109, 255, 255)],
    "red": [[(0, 161, 34), (23, 255, 255)], [(0, 161, 34), (250, 255, 255)]],
    # "blue": [(34, 131, 49), (255, 255, 255)],
    # "blue": [(30, 23, 49), (120, 154, 203)],
    "blue": [(23, 49, 8), (105, 161, 161)],
    "black": [(0, 0, 0), (255, 255, 36)],
}

# сonstants
planewidth = 1200
planeheight = 800
dist_to_plane = 2200  # стандартное?
min_accur = 50
square_hcounts, square_wcounts = 4, 6
size_plus = 150
cross2squareK: float = 200 / 150  # 20 / 15
square_cm = 20

pix_k = 1
floor_plus = 58.5  # real (in cm) distance between the floor (laser level) and plus
floor_bottom = 58.5  # real (in cm) distance between the floor (laser level) and bottom of the plane

try:
    cal_k = float(
        inputimeout("Enter CAL_K value: ", timeout=2) or "15402.0" if not (environ.get('CAL_K')) else environ.get(
            'CAL_K'))  # 1.0
except TimeoutOccurred:
    cal_k = 15402.0


def show(iimg, wd=400):
    h, w = img.shape[:2]
    ratio = w / h
    iimg = cv2.resize(iimg, (wd, round(wd / ratio)), interpolation=cv2.INTER_LINEAR)
    # img = cv2.cvtColor(iimg, cv2.COLOR_HSV2BGR)
    # TODO: HERE DISABLE/ENABLE
    cv2.imshow(str(wd), iimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image = cv2.imread(workpath + "rect5.png")  # 1223.jpg 4438.jpg photo4.jpg rect4.png


# read frame from cam
cam = cv2.VideoCapture(0)
# set cam params
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cam.read()
cam.release()

image = frame


def heq(iimg):
    # We first create a CLAHE model based on OpenCV
    # clipLimit defines threshold to limit the contrast in case of noise in our image
    # tileGridSize defines the area size in which local equalization will be performed
    clahe_model = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(32, 32))

    # For ease of understanding, we explicitly equalize each channel individually
    colorimage_b = clahe_model.apply(iimg[:, :, 0])
    colorimage_g = clahe_model.apply(iimg[:, :, 1])
    colorimage_r = clahe_model.apply(iimg[:, :, 2])

    # Next we stack our equalized channels back into a single image
    return np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)


img = image.copy()

bimg = img.copy()  # Image copy for blue mask creation
bimg = bright_contrast(bimg, 5, 5)  # brightness and contrast
# bimg = heq(bimg)


img = heq(img)
img = bright_contrast(img, 5, 10)  # brightness and contrast

show(image)
show(img)


def draw_col_gist(iimg):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr, _ = np.histogram(iimg[:, :, i], 256, [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show(block=False)
    plt.pause(5)
    plt.close()


draw_col_gist(image)
draw_col_gist(img)

kernel = np.ones((3, 3), np.uint8)  # Generate "drawing" kernel

center_mask = mask_center(img, divk=2)  # Create center mask (half of wd and half of ht), just white rect
center_mask = cv2.cvtColor(center_mask, cv2.COLOR_BGR2GRAY)  # normmalize it

mask_black = mask_by_color(img, color_ranges['black'])  # Shades of black
mask_white = mask_by_color(img, color_ranges['white'])  # Shades of white (and invert)

# -------- PREPARE RED PLUS FOR DETECTION --------

mask_red = mask_by_color(img, color_ranges['red'])  # Shades of red and pink
mask_red = np.where(center_mask == 0, 0, mask_red)  # apply the center mask
mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=1))
mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2))
mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=3))
mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=2))

# mask_red = cv2.subtract(mask_red, mask_black)  # Remove black from red (to avoid crooked plus)
mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2))

print("red mask")
show(mask_red, 500)

# -------- PREPARE RED PLUS FOR DETECTION --------


# -------- PREPARE BLUE MASK --------

mask_blue = mask_by_color(bimg, color_ranges['blue'])  # Shades of blue
# Remove all the red-colored things from blue mask
mask_blue = cv2.subtract(invert(mask_red), invert(mask_blue))
# Close possible holes in the aims
mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2))

# mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=2))  # FIX
mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=2))

# -------- PREPARE BLUE MASK --------

show(mask_blue)

# -------- PREPARE LINES --------

edges = cv2.Canny(img, 50, 200, False)
cop = img.copy()

lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=120,  # Min number of votes for valid line
    minLineLength=35,  # Min allowed length of line
    maxLineGap=15  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    Ax, Ay, Bx, By = points[0]

    dx = Ax - Bx
    dy = Ay - By

    # Draw the lines
    cv2.line(cop, (Ax, Ay), (Bx, By), (0, 255, 0), 2)

    angle = degrees(atan2(dy, dx))
    dist = sqrt((Bx - Ax) ** 2 + (By - Ay) ** 2)
    coords = (Ax, By)

    lines_list.append([coords, dist, angle])

# -------- PREPARE LINES --------


show(cop, 600)

# (lines_list)

# Init cascade
cascade = cv2.CascadeClassifier()
cascade.load(workpath + "cascade2.xml")  # load cascade

# Crop black borders
y_nonzero, x_nonzero = np.nonzero(mask_red)
mask_red_detectable = (mask_red[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])

# Detect cross using cascade
# detected_cross = cascade.detectMultiScale(
#     invert(mask_red_detectable),  # for some reason, it detects black, not white
#     scaleFactor=1.03,
#     minNeighbors=16,
#     minSize=(60, 60),
#     maxSize=(175, 175),
# )

detected_cross = cascade.detectMultiScale(
    invert(mask_red_detectable),  # for some reason, it detects black, not white
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(30, 30),
    maxSize=(220, 220),
)

print(f"Found {len(detected_cross)} objects using haar cascade!")

haar_rect = {"mindist": 0, "center": (0, 0), "wh": (0, 0)}
detects = []

# Iterate over each of detected rect zones
for detect, (dX, dY, dW, dH) in enumerate(detected_cross):
    plus_center = (dX + dW // 2, dY + dH // 2)  # Calculate the center of the rectangle

    # Check if new center is closer to the real center than previous one
    distance = sqrt(plus_center[0] ** 2 + plus_center[1] ** 2)

    # TODO: Check 30 here (too small)
    if (max(dW, dH) / min(dW, dH)) > 1.5 or min(dW, dH) < 30:  # Если соотношение сторон совсем-совмем неправильное
        continue

    temp_dict = dict()
    temp_dict["mindist"] = distance
    temp_dict["center"] = plus_center
    temp_dict["wh"] = (dW, dH)
    temp_dict["xy"] = (dX + np.min(x_nonzero), dY + np.min(y_nonzero))

    if distance < haar_rect["mindist"] or haar_rect["mindist"] == 0:
        # Update the haar_rect dictionary with the new closest rectangle's information
        haar_rect.update(temp_dict)
        detects.insert(0, haar_rect)  # append to the beginning
    else:
        detects.append(temp_dict)  # append to the end

# Draw haar zone rect
for i in detects:
    _ = cv2.rectangle(img,
                      (i["xy"][0], i["xy"][1]),
                      (i["xy"][0] + i["wh"][0], i["xy"][1] + i["wh"][1]),
                      (250, 10, 50),
                      2)
show(mask_red)
show(img)

contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = list(contours)  # convert to list
print(f"Found {len(contours)} contours.")

# Сохраним тех. информацию обо всех контурах
cnts_info = []
for i, cnt in enumerate(contours):
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])  # get center X
        cY = int(M["m01"] / M["m00"])  # get center Y
    else:
        continue

    _, (cW, cH), _ = cv2.minAreaRect(cnt)  # get width and height
    _, _, Rw, Rh = cv2.boundingRect(cnt)  # get real width and height

    if (max(cW, cH) / min(cW, cH)) > 1.5:  # Если соотношение сторон совсем-совмем неправильное
        continue

    # cv2.drawContours(img,[np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))],0,(0,0,255),2) # draw red rotated bounding box

    cnts_info.append({"w": cW, "h": cH, "x": cX, "y": cY, "i": i, "rW": Rw, "rH": Rh})


# pprint(cnts_info)


# Experimental filter of the same "rubbish" countours

def filter_list(lst, places=5):
    result = []

    med_h = statistics.median([d['h'] for d in cnts_info])
    med_w = statistics.median([d['w'] for d in cnts_info])

    places = 2

    rh = range(int(med_h - places), int(med_h + places))
    rw = range(int(med_w - places), int(med_w + places))

    for d in lst:
        if d['w'] not in rw and d['h'] not in rh:
            result.append(d)

    return result


cnts_info = filter_list(cnts_info) if len(cnts_info) > 7 else cnts_info
print(f"Removed {len(contours) - len(cnts_info)} contours totally.")
pprint(cnts_info)

for i in cnts_info:
    cnt = contours[i['i']]
    cv2.drawContours(img, [np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))], 0, (0, 0, 255),
                     2)  # draw red rotated bounding box

show(img, 400)

perfects = []
temp_perfect = []

perfect_ranges = []
temp_perfect_range = []

# Также обхожу каждый элемент (каждую зону)
for detect in detects:
    # Создаём отдельно для каждой зоны диапазоны
    rX = range(
        int((detect["xy"][0])),
        int((detect["xy"][0] + detect["wh"][0])),
    )
    rY = range(
        int((detect["xy"][1])),
        int((detect["xy"][1] + detect["wh"][1])),
    )

    # print(detect["xy"][0], detect["xy"][1])
    # print(rX, rY)
    # print("\n")
    # result = cv2.pointPolygonTest(contour, (x,y), False)
    # Обходим каждый контур
    for cnt in cnts_info:
        cntRx = range(round(cnt['x'] - (cnt['rW'] / 2)), round(cnt['x'] + (cnt['rW'] / 2)))
        cntRy = range(round(cnt['y'] - (cnt['rH'] / 2)), round(cnt['y'] + (cnt['rH'] / 2)))
        if set(cntRx).issubset(rX) and set(cntRy).issubset(rY):
            print("This's our case!", cnt['i'])
            if len(cnts_info) > 1:
                # Каково расстояние до других контуров (если такие есть)
                flag = False
                for other in cnts_info:
                    dist_to_other = sqrt(abs(max(cnt['x'], other['x']) - min(cnt['x'], other['x'])) ** 2 + abs(
                        (max(cnt['x'], other['x']) - min(cnt['x'], other['x']))) ** 2) * 2
                    num_rat = dist_to_other / max(cnt['w'], cnt['h'])
                    print(num_rat)

                    norm_ratio = min(len(rX), len(rY)) / min(len(cntRx), len(cntRy))
                    print("haar zone size to cnt size ratio =", norm_ratio)

                    ideal_plus_ratio_w = round(cnt['rW'] / cnt['w'], 1)
                    ideal_plus_ratio_h = round(cnt['rH'] / cnt['h'], 1)
                    ideal_plus_flag = (ideal_plus_ratio_w == 1.2 or ideal_plus_ratio_h == 1.2)
                    print("Ideal-plus ratio for w:", ideal_plus_ratio_w, "and for h:", ideal_plus_ratio_h)

                    if (num_rat < 25 and num_rat != 0 and norm_ratio < 3) and not ideal_plus_flag:
                        temp_perfect = cnt
                        temp_perfect_range = (rX, rY)
                    elif (num_rat < 25 and num_rat != 0 and norm_ratio < 3) and ideal_plus_flag:
                        flag = True
                    elif num_rat != 0:
                        flag = False
                        temp_perfect = cnt
                        temp_perfect_range = (rX, rY)
                        # break
                    else:
                        continue

                if flag:
                    if cnt not in perfects:
                        perfects.append(cnt)
                        perfect_ranges.append((rX, rY))
                    flag = False
            else:
                temp_perfect = cnt
                temp_perfect_range = (rX, rY)

        if len(cnts_info) <= 1:
            temp_perfect = cnt
            temp_perfect_range = (rX, rY)

if not any(perfects):
    print("tempperfect!!!!")
    perfects.append(temp_perfect)
    perfect_ranges.append(temp_perfect_range)

pprint(perfects)
for ii in perfects:
    cv2.drawContours(img, [contours[ii['i']]], -1, (0, 255, 0), 2)
print(f"Found {len(perfects)} perfect(s)!")
pprint(perfects)
perfect = perfects[-1]
perfect_range = perfect_ranges[-1]

# sizes["plus"] = [perfect['rW'], perfect['rH']]  # save width and height

sizes["plus"] = [max(perfect['rW'], perfect['rH']), max(perfect['rW'], perfect['rH'])]
coordinates["center"] = [round(perfect['x']), round(perfect['y'])]
# save center

show(img, 600)

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

sizes["plane"] = (sizes["square"][0] * square_wcounts, sizes["square"][1] * square_hcounts)

print("Width of the plane: ", sizes["plane"][0])
print("Height of the plane: ", sizes["plane"][1])

center_y = coordinates["center"][1]
center_x = coordinates["center"][0]
plane_h = sizes["plane"][1]
plane_w = sizes["plane"][0]

"""Корректировка поворота картинки по линиям рабочей плоскости"""

limit_x = range(center_x - round(plane_w / 2), center_x + round(plane_w / 2))
limit_y = range(center_y - round(plane_h / 2), center_y + round(plane_h / 2))
print(limit_x)
print(limit_x)

lines_list = [line for line in lines_list if line[0][0] in limit_x and line[0][1] in limit_y]
angles = [angle[2] for angle in lines_list]

# Filter out angles of the vertical lines
angles = [angle for angle in angles if abs(max(abs(angle), 180) - min(abs(angle), 180)) < 30]

print(f"Found {len(lines_list)} on the plane, {len(angles)} of them are horizontal.")

if len(angles) == 0:
    angles = [180, 180, 180]

# TODO: Нужно протестировать. Возможно,  360 - x
medang = statistics.median(angles)
print("Median horizontal angle =", medang)
deltaAng = 360 - min((180 - medang), (180 + medang))
print("delta_ang", deltaAng)

# get rotation matrix around plus center
M = cv2.getRotationMatrix2D((center_x, center_y), deltaAng, 1.0)
img = cv2.warpAffine(img, M, image.shape[1::-1])  # rotate whole image
mask_blue = cv2.warpAffine(mask_blue, M, image.shape[1::-1])  # rotate whole mask_blue

show(img, 500)
# show(rotated, 500)

# print(center_x, center_y)
# print(round(plane_h / 2))
# print(round(plane_w / 2))

# crop img by plane sizes

print("!!!!!!!!!!!", center_y)
img = (img[
       max(round(center_y - (plane_h / 2)), 0): max(round(center_y + (plane_h / 2)), 0),
       max(round(center_x - (plane_w / 2)), 0): max(round(center_x + (plane_w / 2)), 0),
       ])

# show(img)
mask_blue = (mask_blue[
             max(round(center_y - (plane_h / 2)), 0): max(round(center_y + (plane_h / 2)), 0),
             max(round(center_x - (plane_w / 2)), 0): max(round(center_x + (plane_w / 2)), 0),
             ])

# recalculate center coordinates for cropped image
coordinates["center"][0] = coordinates["center"][0] - round(center_x - (plane_w / 2))
coordinates["center"][1] = coordinates["center"][1] - round(center_y - (plane_h / 2))

show(img, 500)
show(mask_blue, 500)

circles = cv2.HoughCircles(
    mask_blue,
    cv2.HOUGH_GRADIENT,
    1.25,
    sizes["square"][0] // 3,
    param1=900,
    param2=5,
    minRadius=round(sizes["square"][0] / 8),
    maxRadius=round(sizes["square"][0] / 5),
)
print("Circles: ", circles)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # TODO: this in new check. Need to be tested
        if center[0] not in perfect_range[0] and center[1] not in perfect_range[1]:
            coordinates["aim_center"].append([i[0], i[1]])
            cv2.circle(img, center, 3, (0, 255, 0), 4)
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 2)

show(img)

centered_coordinates = []
for aim in coordinates['aim_center']:
    # centeredX = coordinates['center'][0] - aim[0]
    # centeredX = centeredX * pix_k * -1
    centeredX = aim[0] * pix_k

    # centeredY = coordinates['center'][1] - aim[1]
    # centeredY = centeredY * pix_k
    centeredY = aim[1] * pix_k

    centered_coordinates.append([centeredX, centeredY])
    print(centeredX, centeredY)

pprint(sizes)


def convert_coordinates(x, y, width, height):
    """
    Function to convert centered coordinates to normal
    """
    half_width = width / 2
    half_height = height / 2
    signs_matcher = {
        -1: 1,
        1: 0,
    }
    x_sign = signs_matcher.get(int(copysign(1, x)))
    y_sign = signs_matcher.get(int(copysign(1, y))) * (-1)
    x_prime = x + half_width + x_sign
    y_prime = (y - half_height) * (-1) + y_sign
    return int(x_prime), int(y_prime)


matrix = [[0] * square_wcounts for i in range(square_hcounts)]
matrix_coords = []
for oldx, oldy in centered_coordinates:
    kx = planewidth / (square_wcounts * 100)
    ky = planeheight / (square_hcounts * 100)

    newx = round((oldx / kx) // 100 + 1)
    newy = round((oldy / ky) // 100 + 1)

    newx = min(square_wcounts, max(0, newx))
    newy = min(square_hcounts, max(0, newy))

    print(newx, newy)
    matrix_coords.append([newx, newy])
    matrix[newy - 1][newx - 1] = 1

pprint(matrix)
pprint(matrix_coords)

# Matrix center: x=3.5, y=2.5
# Мне лень писать код, который будет считать гипотенузы, поэтому я просто запуши их для всех 24 клеток
hypsd = {
    (1, 1): 5,
    (1, 2): 4,
    (1, 3): 4,
    (1, 4): 5,
    (2, 1): 3,
    (2, 2): 2,
    (2, 3): 2,
    (2, 4): 3,
    (3, 1): 2,
    (3, 2): 1,
    (3, 3): 1,
    (3, 4): 2,
    (4, 1): 2,
    (4, 2): 1,
    (4, 3): 1,
    (4, 4): 2,
    (5, 1): 3,
    (5, 2): 2,
    (5, 3): 2,
    (5, 4): 3,
    (6, 1): 5,
    (6, 2): 4,
    (6, 3): 4,
    (6, 4): 5
}

# Сортируем по порядку попадания, как в задании
matrix_coords = sorted(matrix_coords, key=lambda x: (hypsd[tuple(x)], x[0]))

print("Sorted:", matrix_coords)

if len(matrix_coords) > 5:
    matrix_coords = matrix_coords[:5]
    print("Trimmed:", matrix_coords)

plt.figure()
plt.imshow(matrix)
plt.show()

# -------- РЕШЕНИЕ ТРЕУГОЛЬНИКА ПО ВЕРТИКАЛЬНОЙ ОСИ --------
y_coords = [y[1] for y in matrix_coords]
print("y coords", y_coords)

# Базовый угол, на который нужно повернуть серву, чтобы попасть в плюс
base_ang_y = degrees(atan(floor_plus / dist_to_plane))
print("base_ang_y", base_ang_y)
# TODO: ADD BASE_ANG REAL TEST FEATURE

# Считаем гипотенузу (длину луча по Y на базовом угле)
# https://w.wiki/A45H
base_gyp_y = sqrt(floor_plus ** 2 + dist_to_plane ** 2)
print("base_gyp_y", base_gyp_y)

# Считаем угол пересечения луча с холстом (baseline - contact_point - y_plane)
bcy = 180 - (90 - base_ang_y)
print("bcy", bcy)

y_coords = [(y - 2.5) * -1 for y in y_coords]  # back to centerd coords. Again)))
print("y centered coords", y_coords)

y_coords = [y * square_cm for y in y_coords]  # turn to cm
print("y centered coords in cm", y_coords)

ang_y_vals = []

# Решаем треугольник через 2 стороны (y_coords и base_gyp_y) и угол между ними (bcy)
# https://w.wiki/A64E
for dec_y in y_coords:
    # Сначала нужна гипотенуза (длина луча в позиции)
    newgyp = sqrt(dec_y ** 2 + base_gyp_y ** 2 - (2 * dec_y * base_gyp_y * cos(radians(bcy))))
    # Теперь считаем угол, опять по закону косинусов
    ang_y = degrees(acos(((base_gyp_y ** 2 + newgyp ** 2 - dec_y ** 2) / (2 * newgyp * base_gyp_y))))
    # Получилось, но потерялся знак (теперь это модуль угла)
    # Скопируем знак у координаты (centered) по y
    ang_y = copysign(ang_y, dec_y)
    # Это относительный от базового угол
    # Сделаем абсолютным
    ang_y = base_ang_y + ang_y
    ang_y_vals.append(ang_y)
    # print(ang_y)

print(ang_y_vals)

# -------- РЕШЕНИЕ ТРЕУГОЛЬНИКА ПО ВЕРТИКАЛЬНОЙ ОСИ --------


# -------- РЕШЕНИЕ ТРЕУГОЛЬНИКА ПО ГОРИЗОНТАЛЬНОЙ ОСИ --------

# matrix_to_centr = {
#     1: -2.5,
#     2: -1.5,
#     3: -0.5,
#     4: 0.5,
#     5: 1.5,
#     6: 2.5,
# }

# Базовый угол, на который нужно повернуть серву, чтобы попасть в плюс
base_ang_x = 0

x_coords = [x[0] for x in matrix_coords]
print("x coords", x_coords)

# TODO: check this formul for x
x_coords = [(x - 3.5) for x in x_coords]  # back to centerd coords. Again)))
print("x centered coords", x_coords)

x_coords = [x * square_cm for x in x_coords]  # turn to cm
print("x centered coords in cm", x_coords)

ang_x_vals = []

for dec_x in x_coords:
    ang_x = degrees(atan(dec_x / dist_to_plane))
    ang_x = base_ang_x + ang_x
    ang_x_vals.append(ang_x)

print(ang_x_vals)
# -------- РЕШЕНИЕ ТРЕУГОЛЬНИКА ПО ГОРИЗОНТАЛЬНОЙ ОСИ --------

relative_angs = [[ang_x_vals[i], ang_y_vals[i]] for i in range(len(matrix_coords))]
print(relative_angs)

time.sleep(1)

arduino.set_xy(base_ang_x * 4, base_ang_y * 4)
time.sleep(4)

for i in relative_angs:
    arduino.set_xy(i[0] * 4, i[1] * 4)
    time.sleep(4)
