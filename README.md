---
title: Detector de Danos Estructurales
emoji: 🧱
colorFrom: indigo
colorTo: teal
sdk: docker
app_port: 7860
pinned: false
---

# Detector de Daños Estructurales — CNN Ensemble

Aplicación web que pone a producir cuatro redes neuronales convolucionales
(VGG16, ResNet50, MobileNetV2 y EfficientNetB0) entrenadas por transfer
learning para clasificar el estado de una pared o superficie de concreto en
7 categorías: `algae`, `major_crack`, `minor_crack`, `normal`, `peeling`,
`spalling` y `stain`.

El entrenamiento y la comparación de las 4 arquitecturas se hizo en Google
Colab (ver `notebooks/`); este repositorio toma esos modelos ya entrenados y
los pone detrás de una API en Flask con una interfaz web propia, para que el
modelo se pueda usar como un producto real y no solo dentro de un notebook.

## Resultados en el conjunto de prueba (447 imágenes)

| Modelo         | Accuracy | F1-score |
| -------------- | -------- | -------- |
| ResNet50       | 86.35%   | 0.864    |
| VGG16          | 85.23%   | 0.851    |
| MobileNetV2    | 85.23%   | 0.853    |
| EfficientNetB0 | 85.23%   | 0.854    |

La app combina las 4 predicciones promediando sus probabilidades
(*ensemble*) y muestra también el consenso por mayoría de votos.

## Cómo funciona

1. El usuario sube una foto de la superficie desde el navegador.
2. El backend (`app.py`) redimensiona la imagen a 224×224 y la pasa por las
   4 CNN, cada una con su propio preprocesamiento (`preprocess_input`).
3. Se calcula el promedio de las 4 probabilidades por clase (ensemble) y el
   consenso por mayoría de votos.
4. El frontend (`templates/`, `static/`) muestra la predicción conjunta, la
   confianza por clase y la comparación entre las 4 arquitecturas.

## Stack

- **Modelo:** TensorFlow / Keras (transfer learning sobre VGG16, ResNet50,
  MobileNetV2 y EfficientNetB0).
- **Backend:** Python + Flask, servido con Gunicorn.
- **Frontend:** HTML5, CSS3 (custom properties) y JavaScript sin frameworks.
- **Despliegue:** contenedor Docker en Hugging Face Spaces.

## Ejecutar en local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Luego abre `http://localhost:7860`.

## Desplegar en Hugging Face Spaces

Los modelos pesan en total ~360 MB, por lo que necesitan Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git add .
git commit -m "Detector de daños estructurales"
```

Crea un Space nuevo en huggingface.co con SDK **Docker**, agrega su URL como
remoto y súbelo:

```bash
git remote add space https://huggingface.co/spaces/<tu-usuario>/<nombre-del-space>
git push space main
```

El `README.md` ya trae la cabecera que Hugging Face necesita para construir
el Space con Docker en el puerto 7860.
