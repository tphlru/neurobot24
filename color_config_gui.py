#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tools.filters import bright_contrast
from tools.processtool import mask_by_color, mask_center, invert
import numpy as np
from PIL import Image, ImageTk
import cv2
import os
import toml

# Глобальные переменные
current_image = None
current_color = "white"
config_file = "config.toml"
mask_image = None
original_display = None
debug_enabled = False

# Функция для загрузки конфигурации
def load_config():
    if not os.path.exists(config_file):
        messagebox.showerror("Ошибка", f"Файл конфигурации {config_file} не найден")
        return None
    
    try:
        return toml.load(config_file)
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию: {str(e)}")
        return None

# Функция для сохранения конфигурации
def save_config(config):
    try:
        with open(config_file, "w") as f:
            toml.dump(config, f)
        return True
    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию: {str(e)}")
        return False

# Функция для загрузки изображения с камеры
def get_camera_image():
    global current_image
    
    config = load_config()
    if not config:
        return None
    
    status_label.config(text="Загрузка изображения...")
    
    try:
        if config["camera"]["use_file"]:
            # Загружаем изображение из файла, указанного в конфигурации
            file_path = config["camera"]["file_path"]
            if not os.path.exists(file_path):
                status_label.config(text=f"Ошибка: файл {file_path} не найден")
                return None
            
            img = cv2.imread(file_path)
            if img is None:
                status_label.config(text=f"Ошибка: невозможно прочитать файл {file_path}")
                return None
                
            print(f"Загружено изображение из файла: {file_path}")
            print(f"Размер: {img.shape}")
            
            current_image = img
            current_image = bright_contrast(current_image, 
                                           config["image_processing"]["brightness"], 
                                           config["image_processing"]["contrast"])
            process_image()
            status_label.config(text=f"Файл загружен: {file_path}")
            return img
        else:
            # Загружаем с камеры
            try:
                cam = cv2.VideoCapture(config["camera"]["rtsp_url"])
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
                ret, frame = cam.read()
                cam.release()
                
                if ret:
                    print(f"Получено изображение с камеры")
                    print(f"Размер: {frame.shape}")
                    
                    current_image = frame
                    current_image = bright_contrast(current_image, 
                                                  config["image_processing"]["brightness"], 
                                                  config["image_processing"]["contrast"])
                    process_image()
                    status_label.config(text="Изображение с камеры загружено")
                    return frame
                else:
                    status_label.config(text="Не удалось получить изображение с камеры")
                    return None
            except Exception as e:
                status_label.config(text=f"Ошибка камеры: {str(e)}")
                print(f"Ошибка при работе с камерой: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
    except Exception as e:
        status_label.config(text=f"Ошибка загрузки: {str(e)}")
        print(f"Общая ошибка загрузки: {str(e)}")
        import traceback
        traceback.print_exc()
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
        status_label.config(text=f"Загрузка файла: {file_path}...")
        try:
            # Загружаем изображение
            img = cv2.imread(file_path)
            if img is None:
                status_label.config(text=f"Ошибка: невозможно прочитать файл {file_path}")
                return
                
            print(f"Загружено изображение из файла: {file_path}")
            print(f"Размер: {img.shape}")
            
            config = load_config()
            if not config:
                status_label.config(text="Ошибка загрузки конфигурации")
                return
                
            current_image = img
            current_image = bright_contrast(current_image, 
                                          config["image_processing"]["brightness"], 
                                          config["image_processing"]["contrast"])
            
            # Обновляем конфигурацию с новым путем к файлу
            config["camera"]["use_file"] = True
            config["camera"]["file_path"] = file_path
            save_config(config)
            
            process_image()
            status_label.config(text=f"Файл загружен: {os.path.basename(file_path)}")
        except Exception as e:
            status_label.config(text=f"Ошибка загрузки файла: {str(e)}")
            print(f"Ошибка при загрузке файла: {str(e)}")
            import traceback
            traceback.print_exc()

# Функция для автоматической настройки HSV-диапазона
def auto_calibrate_hsv():
    global current_image, current_color
    
    if current_image is None:
        messagebox.showinfo("Информация", "Сначала загрузите изображение")
        return
    
    try:
        # Конвертируем в HSV
        hsv = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        
        # Предопределенные диапазоны для основных цветов
        if current_color == "white":
            # Для белого цвета (низкая насыщенность, высокая яркость)
            return [[0, 0, 180], [255, 30, 255]]
        elif current_color == "black":
            # Для черного цвета (низкая яркость)
            return [[0, 0, 0], [255, 255, 70]]
        elif current_color == "red":
            # Красный цвет имеет два диапазона в пространстве HSV
            return [
                [[0, 70, 50], [10, 255, 255]],
                [[170, 70, 50], [180, 255, 255]]
            ]
        elif current_color == "blue":
            # Синий цвет
            return [[100, 50, 50], [130, 255, 255]]
        else:
            # Пытаемся найти доминирующий цвет в изображении
            # Усредняем HSV по изображению и расширяем диапазон
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            
            # Используем гистограмму для H, чтобы найти доминирующий оттенок
            hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
            max_h_val = np.argmax(hist_h)
            
            # Рассчитываем диапазон: ±20 градусов для оттенка
            h_min = max(0, max_h_val - 20)
            h_max = min(180, max_h_val + 20)
            
            # Для насыщенности и яркости используем значения, типичные для хорошо различимых цветов
            s_min = 50
            s_max = 255
            v_min = 50
            v_max = 255
            
            print(f"Автокалибровка: Доминирующий H={max_h_val}, диапазон H={h_min}-{h_max}")
            
            return [[h_min, s_min, v_min], [h_max, s_max, v_max]]
    
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при автоматической настройке: {str(e)}")
        print(f"Ошибка автокалибровки: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Функция для применения автоматической настройки
def apply_auto_calibration():
    global current_color
    
    # Получаем предлагаемый диапазон
    new_range = auto_calibrate_hsv()
    if not new_range:
        return
    
    config = load_config()
    if not config:
        return
    
    try:
        # Обновляем конфигурацию
        if isinstance(new_range[0][0], list):  # Несколько диапазонов (для красного)
            config["color_ranges"][current_color] = new_range
        else:
            config["color_ranges"][current_color] = new_range
        
        # Сохраняем изменения
        save_config(config)
        
        # Обновляем интерфейс
        update_sliders_from_config()
        update_params_display()
        process_image()
        status_label.config(text=f"Автоматическая настройка для цвета {current_color} применена")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при применении автонастройки: {str(e)}")

# Функция для обработки изображения и создания маски
def process_image():
    global current_image, mask_image, original_display, color_ranges, debug_enabled
    if current_image is None:
        status_label.config(text="Ошибка: изображение не загружено")
        return
        
    config = load_config()
    if not config:
        return
    
    # Сохраняем копию оригинального изображения для отображения
    original_display = current_image.copy()
    
    try:
        # Получаем текущие диапазоны цветов
        color_ranges = config["color_ranges"]
        color_range = color_ranges[current_color]
        
        print(f"Применяем маску для цвета: {current_color}")
        print(f"Диапазон: {color_range}")
        
        # Конвертируем изображение в HSV
        hsv_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)
        print(f"HSV изображение сформировано: {hsv_image.shape}")
        
        # Маскируем изображение в зависимости от типа цветового диапазона
        if isinstance(color_range[0][0], list):  # Несколько диапазонов (например, красный)
            # Создаем маску для каждого диапазона и объединяем
            mask = np.zeros((current_image.shape[0], current_image.shape[1]), dtype=np.uint8)
            for i, subrange in enumerate(color_range):
                print(f"Обработка поддиапазона {i+1}: {subrange}")
                lower = np.array(subrange[0])
                upper = np.array(subrange[1])
                print(f"Нижняя граница: {lower}, тип: {lower.dtype}")
                print(f"Верхняя граница: {upper}, тип: {upper.dtype}")
                
                submask = cv2.inRange(hsv_image, lower, upper)
                print(f"Подмаска создана, ненулевых пикселей: {cv2.countNonZero(submask)}")
                mask = cv2.bitwise_or(mask, submask)
        else:  # Один диапазон
            lower = np.array(color_range[0])
            upper = np.array(color_range[1])
            print(f"Нижняя граница: {lower}, тип: {lower.dtype}")
            print(f"Верхняя граница: {upper}, тип: {upper.dtype}")
            
            mask = cv2.inRange(hsv_image, lower, upper)
        
        # Проверяем, содержит ли маска хоть какие-то значения
        non_zero = cv2.countNonZero(mask)
        print(f"Число ненулевых пикселей в маске: {non_zero}")
        
        # Можно применить морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Если пустая маска, предложить автоматическую калибровку
        if non_zero == 0:
            status_label.config(text="Предупреждение: Маска пуста. Рекомендуется использовать автонастройку")
        
        # Преобразуем маску для отображения
        mask_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Создаем визуализацию результата (маскированное изображение)
        masked_result = cv2.bitwise_and(current_image, current_image, mask=mask)
        
        # Сохраняем маскированное изображение вместо просто маски
        mask_image = masked_result
        
        update_params_display()
        update_image_display()
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при обработке изображения: {str(e)}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

# Функция для обновления отображения изображения с учетом размера окна
def update_image_display():
    global img_label, mask_label, mask_image, original_display
    
    if original_display is None or mask_image is None:
        return
        
    try:
        # Получаем текущий размер frame для правильного масштабирования
        image_width = image_frame.winfo_width() - 20  # Вычитаем отступы
        image_height = image_frame.winfo_height() - 20
        
        # Если размеры еще не определены (первый запуск), используем значения по умолчанию
        if image_width <= 1 or image_height <= 1:
            image_width = 600
            image_height = 600
            
        # Определяем соотношение сторон изображения
        h, w = original_display.shape[:2]
        image_ratio = w / h
        
        # Расчет размеров с сохранением пропорций
        if image_width / image_ratio <= image_height:
            # Ширина является ограничивающим фактором
            display_width = min(image_width // 2, w) 
            display_height = int(display_width / image_ratio)
        else:
            # Высота является ограничивающим фактором
            display_height = min(image_height, h)
            display_width = int(display_height * image_ratio)
            
        # Масштабируем оригинальное изображение
        img_resized = cv2.resize(original_display, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        
        # Масштабируем маскированное изображение
        mask_resized = cv2.resize(mask_image, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        
        # Преобразуем оригинальное изображение для отображения
        b1, g1, r1 = cv2.split(img_resized)
        im1 = Image.fromarray(cv2.merge((r1, g1, b1)))
        imgtk1 = ImageTk.PhotoImage(image=im1)
        img_label.config(image=imgtk1)
        img_label.image = imgtk1
        
        # Преобразуем маскированное изображение для отображения
        b2, g2, r2 = cv2.split(mask_resized)
        im2 = Image.fromarray(cv2.merge((r2, g2, b2)))
        imgtk2 = ImageTk.PhotoImage(image=im2)
        mask_label.config(image=imgtk2)
        mask_label.image = imgtk2
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при обновлении изображения: {str(e)}")
        print(f"Exception in update_image_display: {e}")
        import traceback
        traceback.print_exc()

# Функция для обновления отображения текущих параметров
def update_params_display():
    global current_color
    config = load_config()
    if not config:
        return
    
    color_ranges = config["color_ranges"]
    color_range = color_ranges[current_color]
    
    # Форматируем информацию о диапазонах
    if isinstance(color_range[0][0], list):  # Несколько диапазонов
        params_text = "Цветовые диапазоны:\n"
        for i, subrange in enumerate(color_range):
            params_text += f"Диапазон {i+1}: H:{subrange[0][0]}-{subrange[1][0]}, S:{subrange[0][1]}-{subrange[1][1]}, V:{subrange[0][2]}-{subrange[1][2]}\n"
    else:  # Один диапазон
        params_text = f"Цветовой диапазон: H:{color_range[0][0]}-{color_range[1][0]}, S:{color_range[0][1]}-{color_range[1][1]}, V:{color_range[0][2]}-{color_range[1][2]}"
    
    current_params_label.config(text=params_text)

# Функция для изменения текущего выбранного цвета
def change_color(color):
    global current_color
    current_color = color
    color_label.config(text=f"Текущий цвет: {current_color}")
    update_params_display()
    
    # Обновляем значения слайдеров для текущего цвета
    update_sliders_from_config()
    
    # Обновляем изображение
    process_image()

# Функция для обновления слайдеров на основе конфигурации
def update_sliders_from_config():
    global current_color
    config = load_config()
    if not config:
        return
    
    color_ranges = config["color_ranges"]
    color_range = color_ranges[current_color]
    
    # Очищаем фрейм со слайдерами
    for widget in sliders_container.winfo_children():
        widget.destroy()
    
    if isinstance(color_range[0][0], list):  # Несколько диапазонов
        # Создаем вкладки для каждого диапазона
        notebook = ttk.Notebook(sliders_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        for i, subrange in enumerate(color_range):
            tab = tk.Frame(notebook)
            notebook.add(tab, text=f"Диапазон {i+1}")
            
            # Создаем слайдеры для этого диапазона
            create_sliders_for_range(tab, subrange, i)
    else:  # Один диапазон
        create_sliders_for_range(sliders_container, color_range)

# Функция для создания слайдеров для одного диапазона
def create_sliders_for_range(parent, color_range, range_index=0):
    # Создаем фрейм для минимальных значений
    min_frame = tk.LabelFrame(parent, text="Минимальные значения")
    min_frame.pack(fill=tk.X, pady=5)
    
    # Слайдеры для минимальных значений
    h_min_slider = tk.Scale(min_frame, from_=0, to=255, orient='horizontal', label='H min')
    h_min_slider.set(color_range[0][0])
    h_min_slider.pack(fill=tk.X)
    
    s_min_slider = tk.Scale(min_frame, from_=0, to=255, orient='horizontal', label='S min')
    s_min_slider.set(color_range[0][1])
    s_min_slider.pack(fill=tk.X)
    
    v_min_slider = tk.Scale(min_frame, from_=0, to=255, orient='horizontal', label='V min')
    v_min_slider.set(color_range[0][2])
    v_min_slider.pack(fill=tk.X)
    
    # Создаем фрейм для максимальных значений
    max_frame = tk.LabelFrame(parent, text="Максимальные значения")
    max_frame.pack(fill=tk.X, pady=5)
    
    # Слайдеры для максимальных значений
    h_max_slider = tk.Scale(max_frame, from_=0, to=255, orient='horizontal', label='H max')
    h_max_slider.set(color_range[1][0])
    h_max_slider.pack(fill=tk.X)
    
    s_max_slider = tk.Scale(max_frame, from_=0, to=255, orient='horizontal', label='S max')
    s_max_slider.set(color_range[1][1])
    s_max_slider.pack(fill=tk.X)
    
    v_max_slider = tk.Scale(max_frame, from_=0, to=255, orient='horizontal', label='V max')
    v_max_slider.set(color_range[1][2])
    v_max_slider.pack(fill=tk.X)
    
    # Кнопка для применения изменений
    apply_button = tk.Button(parent, text='Применить изменения', 
                           command=lambda: apply_changes(
                               h_min_slider.get(), s_min_slider.get(), v_min_slider.get(),
                               h_max_slider.get(), s_max_slider.get(), v_max_slider.get(), 
                               range_index))
    apply_button.pack(fill=tk.X, pady=5)

# Функция для применения изменений
def apply_changes(h_min, s_min, v_min, h_max, s_max, v_max, range_index=0):
    global current_color
    config = load_config()
    if not config:
        return
    
    # Обновляем значения в конфигурации
    color_ranges = config["color_ranges"]
    color_range = color_ranges[current_color]
    
    if isinstance(color_range[0][0], list):  # Несколько диапазонов
        # Обновляем указанный диапазон
        color_ranges[current_color][range_index] = [[h_min, s_min, v_min], [h_max, s_max, v_max]]
    else:  # Один диапазон
        color_ranges[current_color] = [[h_min, s_min, v_min], [h_max, s_max, v_max]]
    
    # Сохраняем изменения
    save_config(config)
    
    # Обновляем отображение и обрабатываем изображение
    update_params_display()
    process_image()
    status_label.config(text=f"Параметры для цвета {current_color} обновлены")

# Функция для разделения красного диапазона
def split_red_range():
    config = load_config()
    if not config:
        return
    
    # Проверяем, является ли текущий цвет красным
    if current_color != "red":
        messagebox.showinfo("Информация", "Эта функция доступна только для красного цвета")
        return
    
    # Проверяем, является ли текущий диапазон единичным
    if not isinstance(config["color_ranges"]["red"][0][0], list):
        # Преобразуем в два диапазона
        current_range = config["color_ranges"]["red"]
        h_min, s_min, v_min = current_range[0]
        h_max, s_max, v_max = current_range[1]
        
        # Создаем два диапазона для красного цвета
        config["color_ranges"]["red"] = [
            [[0, s_min, v_min], [20, s_max, v_max]],
            [[160, s_min, v_min], [180, s_max, v_max]]
        ]
        
        # Сохраняем изменения
        save_config(config)
        
        # Обновляем интерфейс
        update_sliders_from_config()
        update_params_display()
        process_image()
        status_label.config(text="Красный цвет разделен на два диапазона")
    else:
        messagebox.showinfo("Информация", "Красный цвет уже разделен на несколько диапазонов")

# Функция для объединения красного диапазона
def merge_red_range():
    config = load_config()
    if not config:
        return
    
    # Проверяем, является ли текущий цвет красным
    if current_color != "red":
        messagebox.showinfo("Информация", "Эта функция доступна только для красного цвета")
        return
    
    # Проверяем, является ли текущий диапазон множественным
    if isinstance(config["color_ranges"]["red"][0][0], list):
        # Берем средние значения S и V из всех диапазонов
        ranges = config["color_ranges"]["red"]
        s_min = min([r[0][1] for r in ranges])
        v_min = min([r[0][2] for r in ranges])
        s_max = max([r[1][1] for r in ranges])
        v_max = max([r[1][2] for r in ranges])
        
        # Создаем единый диапазон для красного цвета
        config["color_ranges"]["red"] = [[0, s_min, v_min], [180, s_max, v_max]]
        
        # Сохраняем изменения
        save_config(config)
        
        # Обновляем интерфейс
        update_sliders_from_config()
        update_params_display()
        process_image()
        status_label.config(text="Красный цвет объединен в один диапазон")
    else:
        messagebox.showinfo("Информация", "Красный цвет уже представлен одним диапазоном")

# Функция для обработки изменения размера окна
def on_resize(event):
    # Обновляем отображение изображения при изменении размера окна
    if mask_image is not None:
        update_image_display()

# Создаем главное окно
root = tk.Tk()
root.title("Настройка цветовых диапазонов HSV")

# Создаем основные фреймы для горизонтального разделения
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

left_frame = tk.Frame(main_frame, width=400)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

right_frame = tk.Frame(main_frame)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# Создаем фрейм для кнопок управления источником в левой панели
source_frame = tk.Frame(left_frame)
source_frame.pack(fill=tk.X, pady=5)

# Кнопки для выбора источника
camera_button = tk.Button(source_frame, text="Использовать камеру/файл из конфига", command=get_camera_image)
camera_button.pack(side=tk.LEFT, padx=5)

file_button = tk.Button(source_frame, text="Загрузить изображение", command=load_image_file)
file_button.pack(side=tk.LEFT, padx=5)

# Фрейм для отображения текущих параметров
params_frame = tk.Frame(left_frame)
params_frame.pack(fill=tk.X, pady=5)

current_params_label = tk.Label(params_frame, text="Загрузите изображение", 
                              font=("Arial", 10, "bold"), justify=tk.LEFT)
current_params_label.pack(side=tk.LEFT, padx=5)

# Фрейм для выбора цвета
color_frame = tk.Frame(left_frame)
color_frame.pack(fill=tk.X, pady=5)

color_label = tk.Label(color_frame, text="Текущий цвет: white")
color_label.pack(side=tk.TOP, padx=5, pady=2)

colors_buttons_frame = tk.Frame(color_frame)
colors_buttons_frame.pack(fill=tk.X)

white_button = tk.Button(colors_buttons_frame, text="Белый", bg="white", fg="black",
                       command=lambda: change_color("white"))
white_button.pack(side=tk.LEFT, padx=5)

red_button = tk.Button(colors_buttons_frame, text="Красный", bg="red", fg="white",
                      command=lambda: change_color("red"))
red_button.pack(side=tk.LEFT, padx=5)

blue_button = tk.Button(colors_buttons_frame, text="Синий", bg="blue", fg="white",
                       command=lambda: change_color("blue"))
blue_button.pack(side=tk.LEFT, padx=5)

black_button = tk.Button(colors_buttons_frame, text="Черный", bg="black", fg="white",
                        command=lambda: change_color("black"))
black_button.pack(side=tk.LEFT, padx=5)

# Дополнительные кнопки для работы с красным цветом
red_options_frame = tk.Frame(left_frame)
red_options_frame.pack(fill=tk.X, pady=5)

split_red_button = tk.Button(red_options_frame, text="Разделить красный", 
                           command=split_red_range)
split_red_button.pack(side=tk.LEFT, padx=5)

merge_red_button = tk.Button(red_options_frame, text="Объединить красный", 
                           command=merge_red_range)
merge_red_button.pack(side=tk.LEFT, padx=5)

# Кнопка автоматической настройки
auto_calibrate_frame = tk.Frame(left_frame)
auto_calibrate_frame.pack(fill=tk.X, pady=5)

auto_calibrate_button = tk.Button(auto_calibrate_frame, text="Автонастройка цвета", 
                                bg="#77c777", fg="white", font=("Arial", 10, "bold"),
                                command=apply_auto_calibration)
auto_calibrate_button.pack(fill=tk.X, padx=5, pady=5)

# Фрейм для слайдеров
sliders_frame = tk.LabelFrame(left_frame, text="Настройки HSV")
sliders_frame.pack(fill=tk.BOTH, expand=True, pady=5)

# Контейнер для слайдеров, который будет очищаться и заполняться заново
sliders_container = tk.Frame(sliders_frame)
sliders_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Изображение в правой панели
image_frame = tk.LabelFrame(right_frame, text="Результат маскирования")
image_frame.pack(fill=tk.BOTH, expand=True)

# Создаем горизонтальный контейнер для двух изображений
images_container = tk.Frame(image_frame)
images_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Левое изображение (оригинал)
left_img_frame = tk.LabelFrame(images_container, text="Оригинал")
left_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

newimg = tk.PhotoImage()
img_label = tk.Label(left_img_frame, image=newimg)
img_label.pack(fill=tk.BOTH, expand=True)

# Правое изображение (маска)
right_img_frame = tk.LabelFrame(images_container, text="Маска")
right_img_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

mask_label = tk.Label(right_img_frame, image=newimg)
mask_label.pack(fill=tk.BOTH, expand=True)

# Привязываем обработчик события изменения размера
image_frame.bind("<Configure>", on_resize)

# Метка статуса внизу окна
status_label = tk.Label(root, text="Готов к работе", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X, side=tk.BOTTOM, pady=2)

# Устанавливаем начальный размер окна
root.geometry("1200x800")

# Загружаем изображение после инициализации интерфейса
root.after(100, get_camera_image)

# Запускаем главный цикл
root.mainloop() 