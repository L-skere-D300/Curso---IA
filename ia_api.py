from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import unicodedata

app = Flask(__name__)
CORS(app)

# =========================
# LIMPIEZA DE TEXTO
# =========================
def limpiar_texto(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    correcciones = {
        "picason": "picazon",
        "comeson": "comezon",
        "doler": "dolor",
        "mareos": "mareo",
        "palpitasion": "palpitacion"
    }

    for mal, bien in correcciones.items():
        texto = texto.replace(mal, bien)

    return texto

# =========================
# CARGAR DATASET
# =========================
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip().str.lower()

if "sintoma" not in df.columns or "especialista" not in df.columns:
    raise Exception("El CSV debe tener columnas: sintoma, especialista")

sintomas = df["sintoma"].astype(str).apply(limpiar_texto).tolist()
especialistas = df["especialista"].astype(str).tolist()

# =========================
# MODELO IA
# =========================
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500)
X = vectorizer.fit_transform(sintomas)

modelo = MultinomialNB()
modelo.fit(X, especialistas)

# =========================
# REGLAS
# =========================
def detectar_por_reglas(texto):
    texto = texto.lower()

    if any(p in texto for p in ["piel", "mancha", "picazon", "comezon", "erupcion", "roncha", "irritacion"]):
        return "dermatologo"

    if any(p in texto for p in ["pecho", "corazon", "palpitacion", "presion"]):
        return "cardiologo"

    if any(p in texto for p in ["muela", "diente", "encia", "dental"]):
        return "odontologo"

    if any(p in texto for p in ["cabeza", "migraña", "mareo", "equilibrio"]):
        return "neurologo"

    if any(p in texto for p in ["respirar", "pulmon", "ahogo"]):
        return "neumologo"

    if any(p in texto for p in ["fiebre", "cansancio", "malestar", "debilidad"]):
        return "medico_general"

    return None

# =========================
# URGENCIA
# =========================
def detectar_urgencia(texto):
    texto = texto.lower()

    if any(p in texto for p in ["fuerte", "intenso", "grave", "no puedo"]):
        return "alta"

    if any(p in texto for p in ["dolor", "fiebre", "mareo"]):
        return "media"

    return "baja"

# =========================
# RESPUESTA NATURAL 🔥
# =========================
def generar_respuesta_natural(texto, especialista, urgencia):

    respuesta = []

    # inicio humano
    if "dolor" in texto:
        respuesta.append("Entiendo que estás sintiendo dolor.")
    elif "picazon" in texto or "comezon" in texto:
        respuesta.append("Parece que estás teniendo una molestia en la piel.")
    else:
        respuesta.append("Gracias por contarme cómo te sientes.")

    # especialista
    respuesta.append(f"Por lo que describes, lo más adecuado sería acudir a un {especialista}.")

    # urgencia
    if urgencia == "alta":
        respuesta.append("Este caso podría ser urgente, te recomiendo buscar atención médica lo antes posible.")
    elif urgencia == "media":
        respuesta.append("No parece grave, pero sería recomendable atenderlo pronto.")
    else:
        respuesta.append("Por ahora no parece algo grave, pero observa si hay cambios.")

    # cierre
    respuesta.append("Si los síntomas empeoran, no dudes en acudir a un centro de salud.")

    return " ".join(respuesta)

# =========================
# API
# =========================
@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.json

    if not data or "texto" not in data:
        return jsonify({"error": "No se envió texto"}), 400

    texto_original = data["texto"]
    texto = limpiar_texto(texto_original)

    especialista = detectar_por_reglas(texto)

    if especialista is None:
        X_test = vectorizer.transform([texto])
        especialista = modelo.predict(X_test)[0]

    urgencia = detectar_urgencia(texto)

    mensaje = generar_respuesta_natural(texto, especialista, urgencia)

    return jsonify({
        "mensaje": mensaje,
        "especialista": especialista,
        "urgencia": urgencia
    })

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5000)