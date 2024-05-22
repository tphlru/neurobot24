import tkinter as tk
from tools.filters import bright_contrast  # histogram_adp
from tools.processtool import mask_by_color, mask_center, invert
from PIL import Image, ImageTk
import cv2

hmin, smin, vmin = 49, 34, 20
hmax, smax, vmax = 120, 145, 205

# immmg = cv2.imread("photo2.jpg")

cam = cv2.VideoCapture(0)
# set cam params
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cam.read()
cam.release()

immmg = frame

testset = [(0, 0, 0), (255, 255, 255)]


def print_values():
    global hmin, smin, vmin, hmax, smax, vmax, testset
    hmin = slider1.get()
    smin = slider2.get()
    vmin = slider3.get()
    hmax = slider4.get()
    smax = slider5.get()
    vmax = slider6.get()
    testset = [(hmin, smin, vmin), (hmax, smax, vmax)]
    print(testset)

def update():
    print_values()
    global img_label
    mask = mask_by_color(immmg, testset)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    h, w = mask.shape[:2]
    ratio = w / h
    imask = cv2.resize(mask, (600, round(600 / ratio)), interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(imask)
    im = Image.fromarray(cv2.merge((r, g, b)))
    imgtk = ImageTk.PhotoImage(image=im)
    img_label.configure(image=imgtk, width=600, height=600)
    img_label.image = imgtk

    root.after(300, update)


root = tk.Tk()

slider1 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='h min')
slider1.set(hmin)
slider1.pack()

slider2 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='s min')
slider2.set(smin)
slider2.pack()

slider3 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='v min')
slider3.set(vmin)
slider3.pack()

separator = tk.Frame(root, height=2, bd=1, relief=tk.SUNKEN)
separator.pack(padx=10, pady=5, fill=tk.X)

slider4 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='h max')
slider4.set(hmax)
slider4.pack()

slider5 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='s max')
slider5.set(smax)
slider5.pack()

slider6 = tk.Scale(root, from_=0, to=255, orient='horizontal', label='v max')
slider6.set(vmax)
slider6.pack()

button = tk.Button(root, text='Get Slider Values', command=print_values)
button.pack()

newimg = tk.PhotoImage()
img_label = tk.Label(root, image=newimg, width=600, height=600)
img_label.pack()

root.after(300, update)
root.mainloop()
