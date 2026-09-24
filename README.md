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

La app combina las predicciones de los modelos cargados promediando sus
probabilidades (*ensemble*) y muestra también el consenso por mayoría de
votos.

> **Nota sobre el despliegue:** por defecto la app solo carga MobileNetV2 y
> EfficientNetB0 (los dos modelos más livianos, ~35 MB en total) para poder
> correr en hosting gratuito con poca RAM. Los 4 modelos completos, con su
> comparación de accuracy, están documentados en `notebooks/4cnn.ipynb` y se
> pueden activar en la app con la variable de entorno `MODELOS_A_CARGAR`
> (ver más abajo).

## Cómo funciona

1. El usuario sube una foto de la superficie desde el navegador.
2. El backend (`app.py`) redimensiona la imagen a 224×224 y la pasa por cada
   CNN cargada, cada una con su propio preprocesamiento (`preprocess_input`).
3. Se calcula el promedio de las probabilidades por clase (ensemble) y el
   consenso por mayoría de votos.
4. El frontend (`templates/`, `static/`) muestra la predicción conjunta, la
   confianza por clase y la comparación entre modelos.

## Stack

- **Modelo:** TensorFlow / Keras (transfer learning sobre VGG16, ResNet50,
  MobileNetV2 y EfficientNetB0).
- **Backend:** Python + Flask, servido con Gunicorn.
- **Frontend:** HTML5, CSS3 (custom properties) y JavaScript sin frameworks.
- **Despliegue:** contenedor Docker en Render (plan gratuito).

## Ejecutar en local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Luego abre `http://localhost:7860`. Por defecto corre con los 2 modelos
livianos; para probar los 4 completos en tu máquina:

```bash
set MODELOS_A_CARGAR=VGG16,ResNet50,MobileNetV2,EfficientNetB0
python app.py
```

## Desplegar en Render (gratis)

Los modelos usan Git LFS porque pesan varios MB cada uno:

```bash
git lfs install
git lfs track "*.keras"
git add .
git commit -m "Detector de daños estructurales"
git push origin main
```

En [render.com](https://render.com):

1. **New +** → **Web Service** → conecta tu repositorio de GitHub.
2. Render detecta el `Dockerfile` automáticamente (Environment: Docker).
3. Elige el plan **Free**.
4. Deja `MODELOS_A_CARGAR` sin definir (usa el valor por defecto, liviano) y
   despliega.

El servicio gratuito de Render se "duerme" tras 15 minutos sin uso y tarda
unos segundos en despertar en la siguiente visita — es normal en un plan
gratuito.

## Alternativa: Hugging Face Spaces

Hugging Face Spaces también sirve para desplegar este Docker, pero desde
2026 los Spaces con cómputo (Docker o Gradio) requieren el plan PRO
($9/mes). Si en algún momento tienes esa suscripción, basta con crear un
Space con SDK **Docker**, agregar esa cabecera al inicio de este README:

```yaml
---
title: Detector de Danos Estructurales
emoji: 🧱
colorFrom: indigo
colorTo: teal
sdk: docker
app_port: 7860
---
```

y hacer `git push` al remoto del Space.
