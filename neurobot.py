#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import statistics
import os
from math import sqrt, atan2, atan, degrees, cos, radians, acos, copysign
import time
import cv2
import numpy as np
import toml
from matplotlib import pyplot as plt
import argparse

# Импорт локальных модулей
from tools.filters import bright_contrast
from tools.processtool import mask_by_color, mask_center, invert
from tools import arduino

# Глобальные переменные
coordinates = {"aim_center": []}
hyps = []  # список для сохранения гипотез до центра
sizes = {}

def load_config(config_path="config.toml"):
    """Загрузка конфигурации из TOML файла"""
    try:
        return toml.load(config_path)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        raise

def save_config(config, config_path="config.toml"):
    """Сохранение конфигурации в TOML файл"""
    try:
        with open(config_path, "w") as f:
            toml.dump(config, f)
        print(f"Конфигурация сохранена в {config_path}")
    except Exception as e:
        print(f"Ошибка сохранения конфигурации: {e}")

def calibrate_camera(config):
    """Калибровка камеры для определения калибровочного коэффициента"""
    print("\n=== РЕЖИМ КАЛИБРОВКИ ===")
    print("1. Установите камеру на известное расстояние от плюса-мишени")
    print("2. Убедитесь, что плюс-мишень хорошо виден в кадре")
    print("3. Измерьте точное расстояние от камеры до плюса в миллиметрах")
    
    # Получение кадра с камеры
    if config["camera"]["use_file"]:
        image_path = config["camera"]["file_path"]
        print(f"\nИспользую изображение из файла: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение из файла: {image_path}")
            return
    else:
        cam = cv2.VideoCapture(config["camera"]["rtsp_url"])
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
        ret, image = cam.read()
        cam.release()
        if not ret:
            print("Не удалось получить кадр с камеры")
            return

    # Копия изображения для обработки
    img = image.copy()
    
    # Применение обработки изображения
    img = bright_contrast(img, config["image_processing"]["brightness"], 
                         config["image_processing"]["contrast"])
    
    # Создание маски для красного плюса
    mask_red = mask_by_color(img, config["color_ranges"]["red"])
    
    # Морфологические операции для улучшения маски
    kernel = np.ones((config["image_processing"]["kernel_size"], 
                     config["image_processing"]["kernel_size"]), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Поиск контуров
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Плюс-мишень не найден на изображении!")
        return
    
    # Находим самый большой контур (предположительно плюс)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # Отрисовка найденного плюса
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"Size: {w}x{h}px", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Показываем изображение с найденным плюсом
    cv2.imshow("Calibration", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Запрос реального расстояния
    while True:
        try:
            real_distance = float(input("\nВведите реальное расстояние до плюса в миллиметрах: "))
            if real_distance <= 0:
                print("Расстояние должно быть положительным числом!")
                continue
            break
        except ValueError:
            print("Пожалуйста, введите корректное число!")
    
    # Вычисление калибровочного коэффициента
    avg_size = (w + h) / 2
    cal_k = real_distance * avg_size
    
    print(f"\nРезультаты калибровки:")
    print(f"Размер плюса на изображении: {avg_size:.1f} пикселей")
    print(f"Реальное расстояние: {real_distance} мм")
    print(f"Вычисленный калибровочный коэффициент (cal_k): {cal_k:.1f}")
    
    # Обновление конфигурации
    config["constants"]["cal_k"] = cal_k
    save_config(config)
    
    print("\nКалибровка завершена! Новый коэффициент сохранен в конфигурации.")
    return cal_k

def debug_show(iimg, wd=400, title=None, debug_enabled=False):
    """Отображение изображения с OpenCV только в режиме отладки"""
    if not debug_enabled:
        return
        
    h, w = iimg.shape[:2]
    ratio = w / h
    iimg = cv2.resize(iimg, (wd, round(wd / ratio)), interpolation=cv2.INTER_LINEAR)
    
    window_name = title if title else str(wd)
    cv2.imshow(window_name, iimg)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def debug_save(iimg, filename, debug_config):
    """Сохранение отладочного изображения"""
    if not debug_config["enabled"] or not debug_config["save_images"]:
        return
        
    output_dir = debug_config["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    file_path = os.path.join(output_dir, filename)
    cv2.imwrite(file_path, iimg)
    print(f"Сохранено отладочное изображение: {file_path}")

def draw_col_gist(iimg, debug_enabled=False):
    """Отрисовка цветовой гистограммы фотографии"""
    if not debug_enabled:
        return
        
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr, _ = np.histogram(iimg[:, :, i], 256, [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show(block=False)
    plt.pause(5)
    plt.close()

def filter_list(lst, places=5):
    """Фильтрация списка контуров для удаления похожих"""
    result = []
    
    # Вычисляем медианные значения высоты и ширины
    med_h = statistics.median([d['h'] for d in lst])
    med_w = statistics.median([d['w'] for d in lst])
    
    # Определяем допустимый диапазон отклонений
    places = 2
    
    # Создаем диапазоны допустимых значений
    rh = range(int(med_h - places), int(med_h + places))
    rw = range(int(med_w - places), int(med_w + places))
    
    # Фильтруем контуры, оставляя только те, которые выходят за пределы допустимых диапазонов
    for d in lst:
        if d['w'] not in rw and d['h'] not in rh:
            result.append(d)
    
    return result

def convert_coordinates(x, y, width, height):
    """Преобразование центрированных координат в нормальные"""
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

def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Neurobot24 - система управления роботом')
    parser.add_argument('--calibrate', action='store_true', help='Запустить режим калибровки')
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config()
    
    # Если запущен режим калибровки
    if args.calibrate:
        calibrate_camera(config)
        return
    
    # Получение настроек отладки
    debug_enabled = config["debug"]["enabled"]
    debug_config = config["debug"]
    
    # Инициализация Arduino
    ard = arduino.Arduino(serport=config["arduino"]["port"], 
                        baud=config["arduino"]["baud"])  # Используем режим эмуляции для тестирования
    # ard.init_arduino()
    
    constants = config["constants"]
    color_ranges = config["color_ranges"]
    
    # Инициализация камеры и получение кадра
    if config["camera"]["use_file"]:
        # Использовать файл вместо камеры
        image_path = config["camera"]["file_path"]
        print(f"Использую изображение из файла: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Не удалось загрузить изображение из файла: {image_path}")
            return
    else:
        # Использовать камеру
        cam = cv2.VideoCapture(config["camera"]["rtsp_url"])
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])
        ret, image = cam.read()
        cam.release()
        
        if not ret:
            print("Не удалось получить кадр с камеры")
            return
    
    # Копии исходного изображения
    img = image.copy()
    bimg = img.copy()  # Копия для создания маски синего цвета
    
    # Применение обработки изображения
    img = bright_contrast(img, config["image_processing"]["brightness"], 
                          config["image_processing"]["contrast"])
    bimg = bright_contrast(bimg, config["image_processing"]["brightness"], 
                           config["image_processing"]["contrast"])
    
    debug_show(image, title="Исходное изображение", debug_enabled=debug_enabled)
    debug_show(img, title="После коррекции яркости/контраста", debug_enabled=debug_enabled)
    debug_save(image, "original.jpg", debug_config)
    debug_save(img, "brightness_contrast.jpg", debug_config)
    
    # Вывод гистограмм
    draw_col_gist(image, debug_enabled=debug_enabled)
    draw_col_gist(img, debug_enabled=debug_enabled)
    
    # Создание ядра для морфологических операций
    kernel_size = config["image_processing"]["kernel_size"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Создание масок
    center_mask = mask_center(img, divk=2)  # Создание центральной маски (половина ширины и высоты)
    center_mask = cv2.cvtColor(center_mask, cv2.COLOR_BGR2GRAY)  # нормализация
    
    mask_black = mask_by_color(img, color_ranges['black'])  # Оттенки черного
    mask_white = mask_by_color(img, color_ranges['white'])  # Оттенки белого
    
    # -------- ПОДГОТОВКА КРАСНОГО ПЛЮСА ДЛЯ ОБНАРУЖЕНИЯ --------
    mask_red = mask_by_color(img, color_ranges['red'])  # Оттенки красного и розового
    mask_red = np.where(center_mask == 0, 0, mask_red)  # Применение центральной маски
    mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=1))
    mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1))
    mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=3))
    mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=2))
    mask_red = (cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2))
    
    print("Red mask")
    debug_show(mask_red, 500, title="Red mask", debug_enabled=debug_enabled)
    debug_save(mask_red, "mask_red.jpg", debug_config)
    
    # -------- ПОДГОТОВКА СИНЕЙ МАСКИ --------
    mask_blue = mask_by_color(bimg, color_ranges['blue'])  # Оттенки синего
    # Удаление всех красных элементов из синей маски
    mask_blue = cv2.subtract(invert(mask_red), invert(mask_blue))
    # Закрытие возможных отверстий в целях
    mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=1))
    mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=3))
    mask_blue = (cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=2))
    
    debug_show(mask_blue, title="Blue mask", debug_enabled=debug_enabled)
    debug_save(mask_blue, "mask_blue.jpg", debug_config)
    
    # -------- ПОДГОТОВКА ЛИНИЙ ДЛЯ ОПРЕДЕЛЕНИЯ УГЛА ПОВОРОТА --------
    edges = cv2.Canny(img, 50, 200, False)
    cop = img.copy()
    
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Входное изображение с границами
        1,  # Разрешение расстояния в пикселях
        np.pi / 180,  # Угловое разрешение в радианах
        threshold=120,  # Минимальное количество голосов для валидной линии
        minLineLength=35,  # Минимальная допустимая длина линии
        maxLineGap=15  # Максимальный допустимый разрыв между линиями для их объединения
    )
    
    # Обход точек
    for points in lines:
        Ax, Ay, Bx, By = points[0]
        
        dx = Ax - Bx
        dy = Ay - By
        
        # Отрисовка линий
        cv2.line(cop, (Ax, Ay), (Bx, By), (0, 255, 0), 2)
        
        angle = degrees(atan2(dy, dx))
        dist = sqrt((Bx - Ax) ** 2 + (By - Ay) ** 2)
        coords = (Ax, By)
        
        lines_list.append([coords, dist, angle])
    
    debug_show(cop, 600, title="Found lines", debug_enabled=debug_enabled)
    debug_save(cop, "lines.jpg", debug_config)
    
    # -------- ДЕТЕКЦИЯ ПО КАСКАДУ ХААРА --------
    # Загрузка классификатора
    cascade = cv2.CascadeClassifier()
    cascade.load(config["haar_cascade"]["filepath"])
    
    # Обрезка черных границ маски для лучшего обнаружения
    y_nonzero, x_nonzero = np.nonzero(mask_red)
    mask_red_detectable = (mask_red[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)])
    
    # Поиск хорошего обнаружения с различными параметрами
    # Это подход методом проб и ошибок, может потребоваться оптимизация
    detected_cross = []
    
    for ifactor in range(1, 200):
        for inei in range(1, 22):
            detected_cross = cascade.detectMultiScale(
                invert(mask_red_detectable),  # по какой-то причине обнаруживает черный, а не белый
                scaleFactor=float(float(ifactor / 1000) + 1),
                minNeighbors=inei,
                minSize=tuple(config["haar_cascade"]["min_size"]),
                maxSize=(350, 350),
            )
            if debug_enabled:
                print("factor =", str(float(ifactor)), "neighbors =", inei, "detected", len(detected_cross))
            if 6 > len(detected_cross) >= 1:
                break
        if 6 > len(detected_cross) >= 1:
            break
            
    print(f"Found {len(detected_cross)} objects using Haar!")
    
    haar_rect = {"mindist": 0, "center": (0, 0), "wh": (0, 0)}
    detects = []
    
    # Обработка обнаруженных прямоугольников
    for detect, (dX, dY, dW, dH) in enumerate(detected_cross):
        plus_center = (dX + dW / 2, dY + dH // 2)  # Вычисление центра прямоугольника
        
        # Проверка, находится ли новый центр ближе к реальному центру, чем предыдущий
        distance = sqrt(plus_center[0] ** 2 + plus_center[1] ** 2)
        
        # Пропуск прямоугольников с неправильным соотношением сторон
        if (max(dW, dH) / min(dW, dH)) > 1.5 or min(dW, dH) < 30:
            continue
        
        temp_dict = dict()
        temp_dict["mindist"] = distance
        temp_dict["center"] = plus_center
        temp_dict["wh"] = (dW, dH)
        temp_dict["xy"] = (dX + np.min(x_nonzero), dY + np.min(y_nonzero))
        
        if distance < haar_rect["mindist"] or haar_rect["mindist"] == 0:
            haar_rect.update(temp_dict)
            detects.insert(0, haar_rect)  # добавить в начало
        else:
            detects.append(temp_dict)  # добавить в конец
    
    # Отрисовка прямоугольников
    for i in detects:
        cv2.rectangle(img,
                     (i["xy"][0], i["xy"][1]),
                     (i["xy"][0] + i["wh"][0], i["xy"][1] + i["wh"][1]),
                     (250, 10, 50),
                     2)
                     
    debug_show(mask_red, title="Red mask for Haar", debug_enabled=debug_enabled)
    debug_show(img, title="Detected objects", debug_enabled=debug_enabled)
    debug_save(img, "detected_objects.jpg", debug_config)
    
    # -------- ДЕТЕКЦИЯ КОНТУРОВ --------
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)  # преобразование в список
    print(f"Найдено {len(contours)} контуров.")
    
    # Получение информации о контурах
    cnts_info = []
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])  # получить центр X
            cY = int(M["m01"] / M["m00"])  # получить центр Y
        else:
            continue
        
        _, (cW, cH), _ = cv2.minAreaRect(cnt)  # получить ширину и высоту
        _, _, Rw, Rh = cv2.boundingRect(cnt)  # получить реальную ширину и высоту
        
        if (max(cW, cH) / min(cW, cH)) > 1.5:  # Пропуск контуров с неправильным соотношением сторон
            continue
        
        cnts_info.append({"w": cW, "h": cH, "x": cX, "y": cY, "i": i, "rW": Rw, "rH": Rh})
    
    # Фильтрация похожих контуров
    cnts_info = filter_list(cnts_info) if len(cnts_info) > 7 else cnts_info
    print(f"Удалено {len(contours) - len(cnts_info)} контуров всего.")
    
    # Отрисовка контуров
    for i in cnts_info:
        cnt = contours[i['i']]
        cv2.drawContours(img, [np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))], 0, (0, 0, 255), 2)
    
    debug_show(img, 400, title="Contours", debug_enabled=debug_enabled)
    debug_save(img, "contours.jpg", debug_config)
    
    # -------- ПОИСК ИДЕАЛЬНЫХ КОНТУРОВ --------
    perfects = []
    temp_perfect = []
    
    perfect_ranges = []
    temp_perfect_range = []
    
    # Сопоставление детекций Хаара с контурами
    for detect in detects:
        rX = range(
            int((detect["xy"][0])),
            int((detect["xy"][0] + detect["wh"][0])),
        )
        rY = range(
            int((detect["xy"][1])),
            int((detect["xy"][1] + detect["wh"][1])),
        )
        
        for cnt in cnts_info:
            cntRx = range(round(cnt['x']), round(cnt['x'] + (cnt['rW'] / 2)))
            cntRy = range(round(cnt['y']), round(cnt['y'] + (cnt['rH'] / 2)))
            if set(cntRx).issubset(rX) and set(cntRy).issubset(rY):
                print("This is our case!", cnt['i'])
                if len(cnts_info) > 1:
                    # Сравнение с другими контурами
                    flag = False
                    for other in cnts_info:
                        dist_to_other = sqrt(abs(max(cnt['x'], other['x']) - min(cnt['x'], other['x'])) ** 2 + abs(
                            (max(cnt['x'], other['x']) - min(cnt['x'], other['x']))) ** 2) * 2
                        num_rat = dist_to_other / max(cnt['w'], cnt['h'])
                        if debug_enabled:
                            print(num_rat)
                        
                        norm_ratio = min(len(rX), len(rY)) / min(len(cntRx), len(cntRy))
                        if debug_enabled:
                            print("соотношение размера зоны хаара к размеру контура =", norm_ratio)
                        
                        ideal_plus_ratio_w = round(cnt['rW'] / cnt['w'], 1)
                        ideal_plus_ratio_h = round(cnt['rH'] / cnt['h'], 1)
                        ideal_plus_flag = (ideal_plus_ratio_w == 1.2 or ideal_plus_ratio_h == 1.2)
                        if debug_enabled:
                            print("Ideal ratio for width:", ideal_plus_ratio_w, "and for height:", ideal_plus_ratio_h)
                        
                        if (num_rat < 25 and num_rat != 0 and norm_ratio < 3) and not ideal_plus_flag:
                            temp_perfect = cnt
                            temp_perfect_range = (rX, rY)
                        elif (num_rat < 25 and num_rat != 0 and norm_ratio < 3) and ideal_plus_flag:
                            flag = True
                        elif num_rat != 0:
                            flag = False
                            temp_perfect = cnt
                            temp_perfect_range = (rX, rY)
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
    
    # Запасной вариант, если не найдены идеальные контуры
    if not any(perfects):
        print("Использую временный совершенный контур в качестве запасного варианта")
        perfects.append(temp_perfect)
        perfect_ranges.append(temp_perfect_range)
    
    if not any(temp_perfect):
        perfects.append(cnts_info[0])
    
    # Используем первый идеальный контур
    for ii in perfects:
        cv2.drawContours(img, [contours[ii['i']]], -1, (0, 255, 0), 2)
    
    print(f"Found {len(perfects)} perfect(s)!")
    perfect = perfects[0]
    perfect_range = perfect_ranges[0]
    
    # Сохраняем размеры плюса
    sizes["plus"] = [max(perfect['rW'], perfect['rH']), max(perfect['rW'], perfect['rH'])]
    coordinates["center"] = [round(perfect['x']), round(perfect['y'])]
    
    debug_show(img, 600, title="Perfect contour", debug_enabled=debug_enabled)
    debug_save(img, "perfect_contour.jpg", debug_config)
    
    # Вычисляем соотношение пикселей к сантиметрам
    print("Ширина плюса: ", sizes["plus"][0])
    print("Высота плюса: ", sizes["plus"][1])
    pix_k = constants["size_plus"] / ((sizes["plus"][0] + sizes["plus"][1]) / 2)
    print("см в 1 пиксель (pix_k): ", pix_k)
    
    # Вычисляем расстояние до плоскости
    dist_to_plane = constants["cal_k"] / ((sizes["plus"][0] + sizes["plus"][1]) / 2)
    print("Расстояние до рабочей плоскости =", dist_to_plane)
    
    # Обновляем конфигурацию новыми значениями
    config["constants"]["dist_to_plane"] = dist_to_plane
    config["constants"]["pix_k"] = pix_k
    
    # Вычисляем размер квадрата на основе размера плюса
    sizes["square"] = (
        int(sizes["plus"][0] * constants["cross2square_k"]),
        int(sizes["plus"][1] * constants["cross2square_k"]),
    )
    
    print("Ширина квадрата: ", sizes["square"][0])
    print("Высота квадрата: ", sizes["square"][1])
    
    # Вычисляем размер плоскости
    sizes["plane"] = (
        sizes["square"][0] * constants["square_wcounts"], 
        sizes["square"][1] * constants["square_hcounts"]
    )
    
    print("Ширина плоскости: ", sizes["plane"][0])
    print("Высота плоскости: ", sizes["plane"][1])
    
    # Получаем координаты центра и размеры плоскости
    center_y = coordinates["center"][1]
    center_x = coordinates["center"][0]
    plane_h = sizes["plane"][1]
    plane_w = sizes["plane"][0]
    
    # -------- КОРРЕКТИРОВКА ВРАЩЕНИЯ ИЗОБРАЖЕНИЯ НА ОСНОВЕ ЛИНИЙ --------
    limit_x = range(center_x - round(plane_w / 2), center_x + round(plane_w / 2))
    limit_y = range(center_y - round(plane_h / 2), center_y + round(plane_h / 2))
    
    # Фильтруем линии, чтобы включать только те, которые находятся на плоскости
    lines_list = [line for line in lines_list if line[0][0] in limit_x and line[0][1] in limit_y]
    angles = [angle[2] for angle in lines_list]
    
    # Фильтруем углы, чтобы включать только горизонтальные линии
    angles = [angle for angle in angles if abs(max(abs(angle), 180) - min(abs(angle), 180)) < 30]
    
    print(f"Найдено {len(lines_list)} линий на плоскости, {len(angles)} из них горизонтальные.")
    
    # Запасной вариант, если не найдены горизонтальные линии
    if len(angles) == 0:
        angles = [180, 180, 180]
    
    # Вычисляем угол вращения
    medang = statistics.median(angles)
    print("Медианный горизонтальный угол =", medang)
    deltaAng = 360 - min((180 - medang), (180 + medang))
    print("delta_ang", deltaAng)
    
    # Вращаем изображение
    M = cv2.getRotationMatrix2D((center_x, center_y), deltaAng, 1.0)
    img = cv2.warpAffine(img, M, image.shape[1::-1])  # вращаем все изображение
    mask_blue = cv2.warpAffine(mask_blue, M, image.shape[1::-1])  # вращаем всю синюю маску
    
    debug_show(img, 500, title="After rotation", debug_enabled=debug_enabled)
    debug_save(img, "rotated.jpg", debug_config)
    
    # Обрезаем изображение до размера плоскости
    img = (img[
           max(round(center_y - (plane_h / 2)), 0): max(round(center_y + (plane_h / 2)), 0),
           max(round(center_x - (plane_w / 2)), 0): max(round(center_x + (plane_w / 2)), 0),
           ])
    
    mask_blue = (mask_blue[
                 max(round(center_y - (plane_h / 2)), 0): max(round(center_y + (plane_h / 2)), 0),
                 max(round(center_x - (plane_w / 2)), 0): max(round(center_x + (plane_w / 2)), 0),
                 ])
    
    # Пересчитываем координаты центра для обрезанного изображения
    coordinates["center"][0] = coordinates["center"][0] - round(center_x - (plane_w / 2))
    coordinates["center"][1] = coordinates["center"][1] - round(center_y - (plane_h / 2))
    
    debug_show(img, 500, title="After cropping", debug_enabled=debug_enabled)
    debug_show(mask_blue, 500, title="Blue mask after cropping", debug_enabled=debug_enabled)
    debug_save(img, "cropped.jpg", debug_config)
    debug_save(mask_blue, "mask_blue_cropped.jpg", debug_config)
    
    # -------- ДЕТЕКЦИЯ КРУГОВ (ЦЕЛЕЙ) --------
    circle_config = config["circle_detection"]
    circles = cv2.HoughCircles(
        mask_blue,
        cv2.HOUGH_GRADIENT,
        circle_config["dp"],
        sizes["square"][0] // circle_config["min_dist_factor"],
        param1=circle_config["param1"],
        param2=circle_config["param2"],
        minRadius=round(sizes["square"][0] / circle_config["min_radius_factor"]),
        maxRadius=round(sizes["square"][0] / circle_config["max_radius_factor"]),
    )
    
    print("Circles: ", circles)
    
    # Обработка обнаруженных кругов
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            if center[0] not in perfect_range[0] and center[1] not in perfect_range[1]:
                coordinates["aim_center"].append([i[0], i[1]])
                cv2.circle(img, center, 3, (0, 255, 0), 4)
                radius = i[2]
                cv2.circle(img, center, radius, (255, 0, 255), 2)
    
    debug_show(img, title="Detected circles", debug_enabled=debug_enabled)
    debug_save(img, "detected_circles.jpg", debug_config)
    
    # -------- РАСЧЕТ КООРДИНАТ ДЛЯ ПРИЦЕЛИВАНИЯ --------
    centered_coordinates = []
    for aim in coordinates['aim_center']:
        centeredX = aim[0] * pix_k
        centeredY = aim[1] * pix_k
        
        centered_coordinates.append([centeredX, centeredY])
        print(centeredX, centeredY)
    
    # Создаем матрицу, представляющую сетку
    matrix = [[0] * constants["square_wcounts"] for i in range(constants["square_hcounts"])]
    matrix_coords = []
    
    for oldx, oldy in centered_coordinates:
        kx = constants["plane_width"] / (constants["square_wcounts"] * 100)
        ky = constants["plane_height"] / (constants["square_hcounts"] * 100)
        
        newx = round((oldx / kx) // 100 + 1)
        newy = round((oldy / ky) // 100 + 1)
        
        newx = min(constants["square_wcounts"], max(0, newx))
        newy = min(constants["square_hcounts"], max(0, newy))
        
        print(newx, newy)
        matrix_coords.append([newx, newy])
        matrix[newy - 1][newx - 1] = 1
    
    # Матрица гипотенуз для каждой ячейки
    # Это определяет порядок, в котором будут поражены цели
    hypsd = {
        (1, 1): 5, (1, 2): 4, (1, 3): 4, (1, 4): 5,
        (2, 1): 3, (2, 2): 2, (2, 3): 2, (2, 4): 3,
        (3, 1): 2, (3, 2): 1, (3, 3): 1, (3, 4): 2,
        (4, 1): 2, (4, 2): 1, (4, 3): 1, (4, 4): 2,
        (5, 1): 3, (5, 2): 2, (5, 3): 2, (5, 4): 3,
        (6, 1): 5, (6, 2): 4, (6, 3): 4, (6, 4): 5
    }
    
    # Сортировка координат по значениям гипотенузы и координате x
    matrix_coords = sorted(matrix_coords, key=lambda x: (hypsd[tuple(x)], x[0]))
    
    print("Sorted:", matrix_coords)
    
    # Ограничение до 5 целей
    if len(matrix_coords) > 5:
        matrix_coords = matrix_coords[:5]
        print("Cropped:", matrix_coords)
    
    # Визуализация матрицы
    if debug_enabled:
        plt.figure()
        plt.imshow(matrix)
        plt.title("Targets matrix")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    
    # -------- РАСЧЕТ УГЛОВ ДЛЯ Y-ОСИ --------
    y_coords = [y[1] for y in matrix_coords]
    print("y coordinates", y_coords)
    
    # Базовый угол для попадания в плюс
    base_ang_y = degrees(atan(constants["floor_plus"] / dist_to_plane))
    print("base_ang_y", base_ang_y)
    
    # Вычисляем длину гипотенузы (длину луча для Y на базовом угле)
    base_gyp_y = sqrt(constants["floor_plus"] ** 2 + dist_to_plane ** 2)
    print("base_gyp_y", base_gyp_y)
    
    # Угол пересечения луча с холстом
    bcy = 180 - (90 - base_ang_y)
    print("bcy", bcy)
    
    # Преобразование в центрированные координаты
    y_coords = [(y - 2.5) * -1 for y in y_coords]
    print("centered y coordinates", y_coords)
    
    # Преобразование в см
    y_coords = [y * constants["square_cm"] for y in y_coords]
    print("centered y coordinates in cm", y_coords)
    
    ang_y_vals = []
    
    # Вычисление Y-углов с использованием закона косинусов
    for dec_y in y_coords:
        # Сначала вычисляем новую длину гипотенузы
        newgyp = sqrt(dec_y ** 2 + base_gyp_y ** 2 - (2 * dec_y * base_gyp_y * cos(radians(bcy))))
        # Вычисляем угол, используя закон косинусов
        ang_y = degrees(acos(((base_gyp_y ** 2 + newgyp ** 2 - dec_y ** 2) / (2 * newgyp * base_gyp_y))))
        # Копируем знак от координаты y
        ang_y = copysign(ang_y, dec_y)
        # Делаем абсолютным от базового угла
        ang_y = base_ang_y + ang_y
        ang_y_vals.append(ang_y)
    
    print(ang_y_vals)
    
    # -------- РАСЧЕТ УГЛОВ ДЛЯ X-ОСИ --------
    base_ang_x = 0
    
    x_coords = [x[0] for x in matrix_coords]
    print("x coordinates", x_coords)
    
    # Преобразование в центрированные координаты
    x_coords = [(x - 3.5) for x in x_coords]
    print("centered x coordinates", x_coords)
    
    # Преобразование в см
    x_coords = [x * constants["square_cm"] for x in x_coords]
    print("centered x coordinates in cm", x_coords)
    
    ang_x_vals = []
    
    # Вычисление X-углов с использованием простой тригонометрии
    for dec_x in x_coords:
        ang_x = degrees(atan(dec_x / dist_to_plane))
        ang_x = base_ang_x + ang_x
        ang_x_vals.append(ang_x)
    
    print(ang_x_vals)
    
    # Объединение X и Y углов
    relative_angs = [[ang_x_vals[i], ang_y_vals[i]] for i in range(len(matrix_coords))]
    print(relative_angs)
    
    # Сохраняем обновленную конфигурацию
    save_config(config)
    
    # Команда Arduino для перемещения к целям
    time.sleep(3)
    
    # for i in relative_angs:
    #     if ard.wait_for_btn():
    #         time.sleep(3)
    #         ard.set_xy(i[0] * 4, i[1] * 4)

if __name__ == "__main__":
    main() 