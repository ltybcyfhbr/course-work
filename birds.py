
GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


import sys
import numpy as np
import cv2
import os
from googlenet_processor import googlenet_processor
from tiny_yolo_processor import tiny_yolo_processor

# будет выполняться на всех изображениях в этом каталоге
input_image_path = './images'

ty_ir= './tiny-yolo-v1_53000.xml'
gn_ir= './googlenet-v1.xml'

# метки для отображения вместе с полями, если классификация GoogleNet хорошая
gn_labels = [""]
cv_window_name = 'Birds - Q to quit or any key to advance'


# Интерпретировать вывод из одного вывода TinyYolo (GetResult)
# и отфильтровать объекты / ящики с низкой вероятностью.
# output - массив чисел с плавающей точкой, возвращаемый из API GetResult, но преобразованный
# в формате float32.
# input_image_width - ширина входного изображения
# input_image_height - высота входного изображения
# Возвращает список списков. каждый из внутренних списков представляет один найденный объект и содержит
# следующие 6 значений:
# 	строка, которая является классификацией сети, то есть «кошка» или «стул» и т. д
# 	значение с плавающей точкой для положения центра поля X в пикселях на исходном изображении
# 	значение с плавающей точкой для положения центра поля Y в пикселях на исходном изображении
# 	float значение ширины блока в пикселях внутри исходного изображения
# 	float значение высоты блока в пикселях внутри исходного изображения
# 	значение с плавающей точкой, которое является вероятностью для классификации сети.

# Отображает окно графического интерфейса с изображением, которое содержит
# коробки и этикетки для найденных предметов. не вернется, пока
# пользователь нажимает клавишу или время ожидания.
# source_image - исходное изображение перед изменением размера или другим способом
#
# Filter_objects - это список списков (как возвращено из filter_objects ()
# а затем добавлен в get_googlenet_classifications ()
# каждый из внутренних списков представляет один найденный объект и содержит
# следующие значения:
# 	строка, которая является классификацией сети yolo, то есть «птица»
# 	значение с плавающей точкой для положения центра поля X в пикселях на исходном изображении
# 	значение с плавающей точкой для положения центра поля Y в пикселях на исходном изображении
# 	float значение ширины блока в пикселях внутри исходного изображения
# 	float значение высоты блока в пикселях внутри исходного изображения
# 	значение с плавающей точкой, которое является вероятностью для классификации yolo.
# 	int значение, которое является индексом классификации googlenet
# 	строковое значение, которое является строкой классификации googlenet.
# 	значение с плавающей точкой, которое является вероятностью googlenet
#
# Возвращает true, если должен перейти к следующему изображению, или false, если
# не должна.
def display_objects_in_gui(source_image, filtered_objects, ty_processor):

    DISPLAY_BOX_WIDTH_PAD = 0
    DISPLAY_BOX_HEIGHT_PAD = 20

    # если googlenet возвращает вероятность меньше этой, то
    # просто используйте tiny yolo более общую классификацию, то есть «птичка»
    GOOGLE_PROB_MIN = 0.5

	# скопировать изображение, чтобы мы могли рисовать на нем.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    x_ratio = float(source_image_width) / ty_processor.ty_w
    y_ratio = float(source_image_height) / ty_processor.ty_h

    # перебрать все поля и нарисовать их на изображении вместе с меткой классификации
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1] * x_ratio)
        center_y = int(filtered_objects[obj_index][2]* y_ratio)
        half_width = int(filtered_objects[obj_index][3]*x_ratio)//2 + DISPLAY_BOX_WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_ratio)//2 + DISPLAY_BOX_HEIGHT_PAD

        # вычислить координаты окна (слева, сверху) и (справа, снизу)
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        # нарисуйте прямоугольник на изображении вокруг объекта
        box_color = (0, 255, 0)  # зеленая коробка
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),(box_right, box_bottom), box_color, box_thickness)

        # нарисовать строку метки классификации чуть выше и слева от прямоугольника
        label_background_color = (70, 120, 70) # серовато-зеленый фон для текста
        label_text_color = (255, 255, 255)   # белый текст

        if (filtered_objects[obj_index][8] > GOOGLE_PROB_MIN):
            label_text = filtered_objects[obj_index][7] + ' : %.2f' % filtered_objects[obj_index][8]
        else:
            label_text = filtered_objects[obj_index][0] + ' : %.2f' % filtered_objects[obj_index][5]

        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = box_left
        label_top = box_top - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]
        cv2.rectangle(display_image,(label_left-1, label_top-1),(label_right+1, label_bottom+1), label_background_color, -1)

        # текст надписи над окном
        cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # отображать текст, чтобы пользователь знал, как выйти
    cv2.rectangle(display_image,(0, 0),(140, 30), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(display_image, "Any key to advance", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow(cv_window_name, display_image)
    raw_key = cv2.waitKey(3000)
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


# Выполняет выводы googlenet для всех объектов, определенных фильтрованными объектами
# Для выполнения выводов будет обрезать изображение из исходного изображения на основе
# поля, определенные в Filter_objects и использовать их в качестве входных данных для GoogleNet
#
#
# source_image исходное изображение, на котором был выполнен вывод. Коробки,
# определенные фильтруемыми объектами, являются прямоугольниками в этом изображении и будут
# используется как вход для googlenet. Это изображение может масштабироваться иначе, чем
# крошечные размеры сети yolo, в этом случае блоки в отфильтрованных объектах
# будет масштабироваться, чтобы соответствовать.
#
# Filter_objects [IN / OUT] при вводе представляет собой список списков (как возвращено из filter_objects ()
#   каждый из внутренних списков представляет один найденный объект и содержит
#    следующие 6 значений:
# 	строка, которая является классификацией сети, то есть «кошка» или «стул» и т. д
# 	значение с плавающей точкой для положения центра поля X в пикселях на исходном изображении
# 	значение с плавающей точкой для положения центра поля Y в пикселях на исходном изображении
# 	float значение ширины блока в пикселях внутри исходного изображения
# 	float значение высоты блока в пикселях внутри исходного изображения
# 	значение с плавающей точкой, которое является вероятностью для классификации сети.
#    после вывода следующих 3 значений из логического вывода googlenet
#     быть добавленным в каждый внутренний список отфильтрованных объектов
# 	int значение, которое является индексом классификации googlenet
# 	строковое значение, которое является строкой классификации googlenet.
# 	значение с плавающей точкой, которое является вероятностью googlenet
#
# возвращаем None
def get_googlenet_classifications(source_image, filtered_objects, gn_processor, ty_processor):

    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]
    x_scale = float(source_image_width) / ty_processor.ty_w
    y_scale = float(source_image_height) / ty_processor.ty_h

    # добавьте высоту и ширину полей изображения на эту величину
    # чтобы убедиться, что мы получаем весь объект на изображении, который
    # переходим в googlenet
    WIDTH_PAD = int(20 * x_scale)
    HEIGHT_PAD = int(30 * y_scale)

    # перебрать все поля и обрезать изображение в этом прямоугольнике
    # из исходного изображения, а затем использовать его в качестве входных данных для Googlenet
    
    for obj_index in range(len(filtered_objects)):
        center_x = int(filtered_objects[obj_index][1]*x_scale)
        center_y = int(filtered_objects[obj_index][2]*y_scale)
        half_width = int(filtered_objects[obj_index][3]*x_scale)//2 + WIDTH_PAD
        half_height = int(filtered_objects[obj_index][4]*y_scale)//2 + HEIGHT_PAD

        # Рассчитать поле (слева, сверху) и (справа, снизу) координаты
        box_left = max(center_x - half_width, 0)
        box_top = max(center_y - half_height, 0)
        box_right = min(center_x + half_width, source_image_width)
        box_bottom = min(center_y + half_height, source_image_height)

        one_image = source_image[box_top:box_bottom, box_left:box_right]
        # Запустите вывод googlenet
        filtered_objects[obj_index] += gn_processor.googlenet_inference(one_image)


    return


# Эта функция вызывается из точки входа, чтобы сделать
# всю работу.
def main():
    global input_image_filename_list
    print('Running NCS birds example')

    # получить список всех файлов .jpg в каталоге изображений
    input_image_filename_list = os.listdir(input_image_path)
    input_image_filename_list = [input_image_path + '/' + i for i in input_image_filename_list if i.endswith('.jpg')]

    if (len(input_image_filename_list) < 1):
        # нет изображений для показа
        print('No .jpg files found')
        return 1

    print('Q to quit, or any key to advance to next image')

    cv2.namedWindow(cv_window_name)

    # Создать Inference Engine Core для внутреннего управления доступными устройствами и их плагинами.
    ie = IECore()
    # Создание процессоров Tiny Yolo и GoogLeNet для выполнения выводов.
    # См. Tiny_yolo_processor.py и googlenet_processor.py для получения дополнительной информации.
    ty_processor = tiny_yolo_processor(ty_ir, ie, DEVICE)
    gn_processor = googlenet_processor(gn_ir, ie, DEVICE)

    for input_image_file in input_image_filename_list :

        # Считать изображение из файла, изменить его размер до ширины и высоты сети
        # сохранить копию в display_image для отображения, затем преобразовать в float32, нормализовать (разделить на 255),
        # и, наконец, конвертировать для преобразования в float16 для передачи в LoadTensor в качестве входных данных для логического вывода
        input_image = cv2.imread(input_image_file)
        
        # изменить размер изображения до стандартной ширины для всех изображений и сохранить соотношение сторон
        STANDARD_RESIZE_WIDTH = 800
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]

        standard_scale = float(STANDARD_RESIZE_WIDTH) / input_image_width
        new_width = int(input_image_width * standard_scale) # это должно быть == STANDARD_RESIZE_WIDTH
        new_height = int(input_image_height * standard_scale)
        input_image = cv2.resize(input_image, (new_width, new_height), cv2.INTER_LINEAR)
        display_image = input_image

        # Запустите tiny yolo и получите список отфильтрованных объектов
        filtered_objs = ty_processor.tiny_yolo_inference(input_image)
        # Передать список отфильтрованных объектов в googlenet для классификации
        get_googlenet_classifications(display_image, filtered_objs, gn_processor, ty_processor)

        # проверить, было ли окно закрыто. все свойства вернут -1.0
        # для закрытых окон. Если пользователь закрыл окно через
        # x в строке заголовка, тогда мы вырвемся из цикла. мы
        # получаем соотношение сторон свойства, но это может быть любое свойство
        # может работать только с opencv 3.x
        try:
            prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        except:
            break
        prop_asp = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_asp < 0.0):
            # возвращенное свойство было <0, поэтому предположим, что окно закрыто пользователем
            break

        ret_val = display_objects_in_gui(display_image, filtered_objs, ty_processor)
        if (not ret_val):
            break

    print(' Finished.')


# главная точка входа в программу. мы будем вызывать main (), чтобы сделать то, что нужно сделать.
if __name__ == "__main__":
    sys.exit(main())
