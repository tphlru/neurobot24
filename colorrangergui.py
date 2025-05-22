import tkinter as tk
from tkinter import filedialog, messagebox
from tools.filters import bright_contrast  # histogram_adp
from tools.processtool import mask_by_color, mask_center, invert
from tools.filters import bright_contrast
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import tomli
import tomli_w

hmin, smin, vmin = 23, 49, 8
hmax, smax, vmax = 105, 161, 161

current_image = None
current_preset_name = "Маска 1"
config_file = "config.toml"
mask_image = None  # Глобальная переменная для хранения маски до масштабирования

# Функция для загрузки изображения с камеры
def get_camera_image():
    global current_image
    cam = cv2.VideoCapture(0)
    # set cam params
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ret, frame = cam.read()
    cam.release()
    
    if ret:
        current_image = frame
        current_image = bright_contrast(current_image, 5, 5)  # brightness and contrast
        process_image()
        status_label.config(text="Изображение с камеры загружено")
        return frame
    return None

# Функция для загрузки изображения из файла
def load_image_file():
    global current_image
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("Все файлы", "*.*")
        ]
    )
    
    if file_path:
        # Загружаем изображение
        img = cv2.imread(file_path)
        if img is not None:
            current_image = img
            current_image = bright_contrast(current_image, 5, 5)  # brightness and contrast
            process_image()
            status_label.config(text=f"Файл загружен: {file_path.split('/')[-1]}")
        else:
            status_label.config(text="Ошибка при загрузке файла")

# Функция для обработки изображения и создания маски
def process_image():
    global current_image, mask_image, testset
    if current_image is not None:
        mask = mask_by_color(current_image, testset)
        mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        update_params_display()
        update_image_display()

# Функция для обновления отображения изображения с учетом размера окна
def update_image_display():
    global img_label, mask_image
    if mask_image is not None:
        # Получаем текущий размер frame для правильного масштабирования
        image_width = image_frame.winfo_width() - 20  # Вычитаем отступы
        image_height = image_frame.winfo_height() - 20
        
        # Если размеры еще не определены (первый запуск), используем значения по умолчанию
        if image_width <= 1 or image_height <= 1:
            image_width = 600
            image_height = 600
            
        # Определяем соотношение сторон изображения
        h, w = mask_image.shape[:2]
        image_ratio = w / h
        
        # Расчет размеров с сохранением пропорций
        if image_width / image_ratio <= image_height:
            # Ширина является ограничивающим фактором
            display_width = min(image_width, w)
            display_height = int(display_width / image_ratio)
        else:
            # Высота является ограничивающим фактором
            display_height = min(image_height, h)
            display_width = int(display_height * image_ratio)
            
        # Масштабируем изображение с учетом доступного пространства
        imask = cv2.resize(mask_image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        
        # Преобразуем изображение для отображения
        b, g, r = cv2.split(imask)
        im = Image.fromarray(cv2.merge((r, g, b)))
        imgtk = ImageTk.PhotoImage(image=im)
        img_label.config(image=imgtk)
        img_label.image = imgtk

# Функции для работы с конфигурациями масок
def load_masks_config():
    if not os.path.exists(config_file):
        # Создаем файл с начальной конфигурацией
        default_config = {
            "Маска 1": {
                "hmin": hmin,
                "smin": smin,
                "vmin": vmin,
                "hmax": hmax,
                "smax": smax,
                "vmax": vmax
            },
            "Маска 2": {
                "hmin": 0,
                "smin": 0,
                "vmin": 0,
                "hmax": 255,
                "smax": 255,
                "vmax": 255
            },
            "Маска 3": {
                "hmin": 0,
                "smin": 0,
                "vmin": 0,
                "hmax": 255,
                "smax": 255,
                "vmax": 255
            }
        }
        save_masks_config(default_config)
        return default_config
    
    try:
        with open(config_file, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию: {str(e)}")
        return {}

def save_masks_config(config=None):
    global hmin, smin, vmin, hmax, smax, vmax, current_preset_name
    
    if config is None:
        # Загружаем текущую конфигурацию, если она существует
        if os.path.exists(config_file):
            try:
                with open(config_file, "rb") as f:
                    config = tomli.load(f)
            except:
                config = {}
        else:
            config = {}
        
        # Обновляем текущую маску
        config[current_preset_name] = {
            "hmin": hmin,
            "smin": smin,
            "vmin": vmin,
            "hmax": hmax,
            "smax": smax,
            "vmax": vmax
        }
    
    try:
        with open(config_file, "wb") as f:
            tomli_w.dump(config, f)
        return True
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию: {str(e)}")
        return False

def apply_mask_preset(preset_name):
    global hmin, smin, vmin, hmax, smax, vmax, current_preset_name, testset
    
    config = load_masks_config()
    if preset_name in config:
        current_preset_name = preset_name
        mask_config = config[preset_name]
        
        hmin = mask_config["hmin"]
        smin = mask_config["smin"]
        vmin = mask_config["vmin"]
        hmax = mask_config["hmax"]
        smax = mask_config["smax"]
        vmax = mask_config["vmax"]
        
        # Обновляем значения слайдеров
        slider1.set(hmin)
        slider2.set(smin)
        slider3.set(vmin)
        slider4.set(hmax)
        slider5.set(smax)
        slider6.set(vmax)
        
        testset = [(hmin, smin, vmin), (hmax, smax, vmax)]
        process_image()
        preset_label.config(text=f"Текущий пресет: {current_preset_name}")
        status_label.config(text=f"Загружен пресет: {current_preset_name}")
        return True
    else:
        status_label.config(text=f"Ошибка: пресет {preset_name} не найден")
        return False

def save_current_preset():
    global current_preset_name
    save_masks_config()
    status_label.config(text=f"Пресет {current_preset_name} сохранен")

# Обновление отображения текущих параметров
def update_params_display():
    params_text = f"H: {hmin}-{hmax}, S: {smin}-{smax}, V: {vmin}-{vmax}"
    current_params_label.config(text=params_text)

# Инициализация с изображением с камеры
try:
    current_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    status_label_text = "Готов к работе"
except:
    # Если произошла ошибка, используем заглушку
    current_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    status_label_text = "Ошибка инициализации"

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
    update_params_display()
    print(testset)

def update():
    print_values()
    if current_image is not None:
        process_image()
    root.after(300, update)

# Функция для обработки изменения размера окна
def on_resize(event):
    # Обновляем отображение изображения при изменении размера окна
    if mask_image is not None:
        update_image_display()

root = tk.Tk()
root.title("Color Range GUI")

# Создаем основные фреймы для горизонтального разделения
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

left_frame = tk.Frame(main_frame, width=300)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Создаем фрейм для кнопок управления источником в левой панели
source_frame = tk.Frame(left_frame)
source_frame.pack(fill=tk.X, pady=5)

# Кнопки для выбора источника
camera_button = tk.Button(source_frame, text="Использовать камеру", command=get_camera_image)
camera_button.pack(side=tk.LEFT, padx=5)

file_button = tk.Button(source_frame, text="Загрузить изображение", command=load_image_file)
file_button.pack(side=tk.LEFT, padx=5)

# Фрейм для отображения текущих параметров
params_frame = tk.Frame(left_frame)
params_frame.pack(fill=tk.X, pady=5)

current_params_label = tk.Label(params_frame, text="H: 0-255, S: 0-255, V: 0-255", 
                                font=("Arial", 10, "bold"))
current_params_label.pack(side=tk.LEFT, padx=5)

# Фрейм для пресетов
presets_frame = tk.Frame(left_frame)
presets_frame.pack(fill=tk.X, pady=5)

preset_label = tk.Label(presets_frame, text="Текущий пресет: Маска 1")
preset_label.pack(side=tk.TOP, padx=5, pady=2)

presets_buttons_frame = tk.Frame(presets_frame)
presets_buttons_frame.pack(fill=tk.X)

mask1_button = tk.Button(presets_buttons_frame, text="Маска 1", 
                        command=lambda: apply_mask_preset("Маска 1"))
mask1_button.pack(side=tk.LEFT, padx=5)

mask2_button = tk.Button(presets_buttons_frame, text="Маска 2", 
                        command=lambda: apply_mask_preset("Маска 2"))
mask2_button.pack(side=tk.LEFT, padx=5)

mask3_button = tk.Button(presets_buttons_frame, text="Маска 3", 
                        command=lambda: apply_mask_preset("Маска 3"))
mask3_button.pack(side=tk.LEFT, padx=5)

save_button = tk.Button(presets_buttons_frame, text="Сохранить", 
                       command=save_current_preset)
save_button.pack(side=tk.LEFT, padx=5)

# Слайдеры в левой панели
slider_frame = tk.LabelFrame(left_frame, text="Настройки HSV")
slider_frame.pack(fill=tk.X, pady=5)

slider1 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='h min')
slider1.set(hmin)
slider1.pack(fill=tk.X)

slider2 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='s min')
slider2.set(smin)
slider2.pack(fill=tk.X)

slider3 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='v min')
slider3.set(vmin)
slider3.pack(fill=tk.X)

separator = tk.Frame(slider_frame, height=2, bd=1, relief=tk.SUNKEN)
separator.pack(padx=5, pady=5, fill=tk.X)

slider4 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='h max')
slider4.set(hmax)
slider4.pack(fill=tk.X)

slider5 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='s max')
slider5.set(smax)
slider5.pack(fill=tk.X)

slider6 = tk.Scale(slider_frame, from_=0, to=255, orient='horizontal', label='v max')
slider6.set(vmax)
slider6.pack(fill=tk.X)

button = tk.Button(left_frame, text='Get Slider Values', command=print_values)
button.pack(fill=tk.X, pady=5)

# Изображение в правой панели
image_frame = tk.LabelFrame(right_frame, text="Результат маскирования")
image_frame.pack(fill=tk.BOTH, expand=True)

newimg = tk.PhotoImage()
img_label = tk.Label(image_frame, image=newimg)
img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Привязываем обработчик события изменения размера
image_frame.bind("<Configure>", on_resize)

# Метка статуса внизу окна
status_label = tk.Label(root, text=status_label_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=2)

# Пытаемся загрузить существующую конфигурацию
try:
    load_masks_config()
    apply_mask_preset("Маска 1")
except:
    pass

# Пытаемся загрузить изображение с камеры после инициализации интерфейса
try:
    get_camera_image()
except:
    pass

# Устанавливаем начальный размер окна
root.geometry("1000x700")

root.after(300, update)
root.mainloop()
