import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_SIZE = (224, 224)

CLASES_ES = {
    "algae": "Algas o moho",
    "major_crack": "Grieta mayor",
    "minor_crack": "Grieta menor",
    "normal": "Sin daño visible",
    "peeling": "Pintura descascarada",
    "spalling": "Desprendimiento de concreto",
    "stain": "Mancha o humedad",
}


def cargar_modelo(nombre_archivo, funcion_preprocesamiento):
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    return tf.keras.models.load_model(
        ruta,
        custom_objects={"preprocess_input": funcion_preprocesamiento},
        compile=False,
        safe_mode=False,
    )


print("Cargando los 4 modelos entrenados, esto puede tardar unos segundos...")
MODELOS = {
    "VGG16": cargar_modelo("mejor_vgg16_final.keras", preprocess_vgg16),
    "ResNet50": cargar_modelo("mejor_resnet50_final.keras", preprocess_resnet50),
    "MobileNetV2": cargar_modelo("mejor_mobilenetv2_fase1.keras", preprocess_mobilenetv2),
    "EfficientNetB0": cargar_modelo("mejor_efficientnetb0_fase1.keras", preprocess_efficientnet),
}

with open(os.path.join(MODELS_DIR, "class_names.json"), encoding="utf-8") as archivo:
    CLASS_NAMES = json.load(archivo)

print("Modelos listos:", ", ".join(MODELOS))

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predecir")
def predecir():
    archivo = request.files.get("imagen")
    if archivo is None or archivo.filename == "":
        return jsonify({"error": "Sube una imagen de la superficie a analizar."}), 400

    try:
        imagen = Image.open(archivo.stream).convert("RGB").resize(IMG_SIZE)
    except Exception:
        return jsonify({"error": "El archivo no es una imagen válida."}), 400

    lote = np.expand_dims(np.asarray(imagen, dtype=np.float32), axis=0)

    resultados = []
    probabilidades_por_modelo = []
    for nombre, modelo in MODELOS.items():
        probabilidades = modelo.predict(lote, verbose=0)[0]
        indice = int(np.argmax(probabilidades))
        clase = CLASS_NAMES[indice]
        resultados.append(
            {
                "modelo": nombre,
                "clase_es": CLASES_ES.get(clase, clase),
                "confianza": round(float(probabilidades[indice]) * 100, 2),
            }
        )
        probabilidades_por_modelo.append(probabilidades)

    promedio = np.mean(probabilidades_por_modelo, axis=0)
    indice_ensamble = int(np.argmax(promedio))
    clase_ensamble = CLASS_NAMES[indice_ensamble]

    votos = {}
    for resultado, clase in zip(resultados, [CLASS_NAMES[int(np.argmax(p))] for p in probabilidades_por_modelo]):
        votos[clase] = votos.get(clase, 0) + 1
    clase_consenso = max(votos, key=votos.get)

    return jsonify(
        {
            "modelos": resultados,
            "ensamble": {
                "clase_es": CLASES_ES.get(clase_ensamble, clase_ensamble),
                "confianza": round(float(promedio[indice_ensamble]) * 100, 2),
                "probabilidades": {
                    CLASES_ES.get(nombre, nombre): round(float(valor) * 100, 2)
                    for nombre, valor in zip(CLASS_NAMES, promedio)
                },
            },
            "consenso": {
                "clase_es": CLASES_ES.get(clase_consenso, clase_consenso),
                "votos": votos[clase_consenso],
                "total": len(resultados),
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
