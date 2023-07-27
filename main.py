import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
# Чтение изображение
img = io.imread('Путь картинки')
# Преобразование изображение и деление пикселей на три столбца RGB
original_shape = img.shape
pixels = img.reshape((-1,3))
print(original_shape)
# Перемешивание пикселей для обучения модели 
pixels_sampels = shuffle(pixels,random_state=0)[:1000]

num_clasters = 4
# Создание и обучение модели
model = KMeans(n_clusters=num_clasters,random_state=42)
model.fit(pixels_sampels)

# Формирование изображения из средних значений кластеров
clasters_label = model.predict(pixels)
clasters_center = model.cluster_centers_
print(clasters_center[clasters_label])
segmente_img = clasters_center[clasters_label].reshape(original_shape)


plt.figure(figsize=(6,6))
plt.imshow(segmente_img.astype(np.uint8))
plt.axis('off')
plt.show()
