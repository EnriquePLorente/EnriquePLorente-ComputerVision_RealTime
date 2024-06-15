import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageDraw, ImageFont


import warnings
warnings.filterwarnings("ignore")

def dibujar_img(img, size=(7,7)):
    fig = plt.figure(figsize=size)
    io.imshow(img)
    io.show()
    
def dibujar_imgs(left, right, size=(10,10)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=size)
    ax[0].imshow(left)
    ax[1].imshow(right)
    io.show()

def reconoce_imagen(im, prob, model):
    # Carga la imagen
    img = Image.fromarray(im, 'RGB')

    # Realiza la predicción
    results = model(im, conf=prob)


    boxes = results[0].boxes.xyxy  # Guarda los borjes de la caja
    confidences = results[0].boxes.conf  # Guarda la probabilidad
    classes = results[0].boxes.cls  # Guarda la clase

    # Crea objeto draw
    draw = ImageDraw.Draw(img)


    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box
        label = model.names[int(cls)] 
        fnt = ImageFont.truetype("arial.ttf", size=30)
        draw.text(((x1 + x2) / 2, y1 + 20), f"{label} {conf:.2f}", font=fnt, fill='black') 
        draw.rectangle([x1, y1, x2, y2], outline=(30,140,100), width=7)


    return img