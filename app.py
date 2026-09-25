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

# Recomendación práctica por tipo de daño: qué hacer y con qué urgencia.
# Pensado para que el encargado de obra sepa cómo priorizar sin necesitar
# todavía una visita de un especialista para los casos leves.
RECOMENDACIONES = {
    "normal": {
        "urgencia": "baja",
        "accion": "No se detectan daños.",
        "detalle": "Continuar con las inspecciones rutinarias (cada 6-12 meses). No se requiere intervención.",
    },
    "minor_crack": {
        "urgencia": "media",
        "accion": "Sellar y monitorear.",
        "detalle": "Suele originarse por retracción del concreto o asentamientos leves. Sellar con masilla elastomérica y volver a fotografiar el mismo punto cada 3 meses para ver si la grieta avanza.",
    },
    "major_crack": {
        "urgencia": "alta",
        "accion": "Requiere evaluación estructural.",
        "detalle": "Puede indicar un problema de fondo (asentamiento, sobrecarga o falla de diseño). Antes de reparar, un ingeniero estructural debe inspeccionar el elemento afectado.",
    },
    "spalling": {
        "urgencia": "alta",
        "accion": "Reparar antes de que avance la corrosión.",
        "detalle": "El desprendimiento suele exponer varillas corroídas. Picar el concreto suelto, tratar el acero con inhibidor de corrosión y resanar con mortero estructural.",
    },
    "peeling": {
        "urgencia": "baja",
        "accion": "Reparación estética, sin urgencia estructural.",
        "detalle": "Remover la pintura suelta, lijar, aplicar sellador y repintar. Si reaparece rápido, revisar si hay una fuente de humedad detrás.",
    },
    "stain": {
        "urgencia": "media",
        "accion": "Buscar el origen de la humedad antes de repintar.",
        "detalle": "La mancha suele ser síntoma de una filtración (tubería, impermeabilización de techo o fachada). Corregir esa fuente primero; repintar sin hacerlo solo oculta el problema.",
    },
    "algae": {
        "urgencia": "media",
        "accion": "Limpiar y mejorar ventilación/drenaje.",
        "detalle": "Indica humedad constante y poca luz solar. Lavar con biocida, revisar el drenaje cercano y aplicar pintura antihongos para evitar que reaparezca.",
    },
}


def cargar_modelo(nombre_archivo, funcion_preprocesamiento):
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    return tf.keras.models.load_model(
        ruta,
        custom_objects={"preprocess_input": funcion_preprocesamiento},
        compile=False,
        safe_mode=False,
    )


CONFIGURACION_MODELOS = {
    "VGG16": ("mejor_vgg16_final.keras", preprocess_vgg16),
    "ResNet50": ("mejor_resnet50_final.keras", preprocess_resnet50),
    "MobileNetV2": ("mejor_mobilenetv2_fase1.keras", preprocess_mobilenetv2),
    "EfficientNetB0": ("mejor_efficientnetb0_fase1.keras", preprocess_efficientnet),
}

# Por defecto solo se cargan los 2 modelos livianos (~35 MB en total) para que
# la app funcione en hosting gratuito con poca RAM (p. ej. Render free, 512 MB).
# Para correr los 4 modelos completos (como en el notebook de entrenamiento),
# define la variable de entorno MODELOS_A_CARGAR=VGG16,ResNet50,MobileNetV2,EfficientNetB0
NOMBRES_MODELOS = os.environ.get("MODELOS_A_CARGAR", "MobileNetV2,EfficientNetB0").split(",")

print(f"Cargando modelos: {', '.join(NOMBRES_MODELOS)}...")
MODELOS = {
    nombre: cargar_modelo(*CONFIGURACION_MODELOS[nombre])
    for nombre in NOMBRES_MODELOS
    if nombre in CONFIGURACION_MODELOS
}

# Los modelos son clasificadores "cerrados": siempre eligen la clase más
# probable de las 7, aunque la imagen no sea una pared (un paisaje, una
# persona, etc.). Para filtrar esos casos exigimos que el ensamble esté
# razonablemente seguro Y que los modelos coincidan entre sí; si no se
# cumple alguna de las dos condiciones, se considera que la imagen no es
# una superficie reconocible en vez de forzar un diagnóstico.
UMBRAL_CONFIANZA_MINIMA = 55.0  # % de confianza del ensamble
UMBRAL_ACUERDO_MINIMO = 0.5  # fracción de modelos que deben coincidir con la clase del ensamble

with open(os.path.join(MODELS_DIR, "class_names.json"), encoding="utf-8") as archivo:
    CLASS_NAMES = json.load(archivo)

print("Modelos listos:", ", ".join(MODELOS))

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", modelos=list(MODELOS.keys()))


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

    confianza_ensamble = round(float(promedio[indice_ensamble]) * 100, 2)
    acuerdo = votos.get(clase_ensamble, 0) / len(resultados)

    if confianza_ensamble < UMBRAL_CONFIANZA_MINIMA or acuerdo < UMBRAL_ACUERDO_MINIMO:
        return jsonify(
            {
                "reconocido": False,
                "mensaje": (
                    "La imagen no parece ser una pared o superficie de concreto. "
                    "Sube una foto clara de la superficie que quieres inspeccionar."
                ),
            }
        )

    return jsonify(
        {
            "reconocido": True,
            "modelos": resultados,
            "ensamble": {
                "clase_es": CLASES_ES.get(clase_ensamble, clase_ensamble),
                "confianza": confianza_ensamble,
                "probabilidades": {
                    CLASES_ES.get(nombre, nombre): round(float(valor) * 100, 2)
                    for nombre, valor in zip(CLASS_NAMES, promedio)
                },
                "recomendacion": RECOMENDACIONES.get(clase_ensamble),
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
