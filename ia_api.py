from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)


# MODELO IA (respaldo)


sintomas = [
    "dolor en el pecho","opresion en el pecho","palpitaciones",
    "dolor de muela","dolor dental","encías inflamadas",
    "manchas en la piel","picazon en la piel","erupciones",
    "dolor de cabeza","migraña","mareos",
    "fiebre","cansancio","malestar general"
]

especialistas = [
    "cardiólogo","cardiólogo","cardiólogo",
    "odontólogo","odontólogo","odontólogo",
    "dermatólogo","dermatólogo","dermatólogo",
    "neurólogo","neurólogo","neurólogo",
    "médico general","médico general","médico general"
]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(sintomas)

modelo = MultinomialNB()
modelo.fit(X, especialistas)

# =========================
# REGLAS INTELIGENTES
# =========================

def detectar_por_reglas(texto):
    texto = texto.lower()

    # CARDIO
    if any(p in texto for p in ["pecho", "corazon", "palpitaciones"]):
        return "cardiólogo"

    # ODONTO
    if any(p in texto for p in ["muela", "diente", "encía", "dental"]):
        return "odontólogo"

    # DERMATO
    if any(p in texto for p in ["piel", "mancha", "picazon", "picazón", "erupcion", "roncha"]):
        return "dermatólogo"

    # NEURO
    if any(p in texto for p in ["cabeza", "migraña", "mareo", "mareos"]):
        return "neurólogo"

    # GENERAL
    if any(p in texto for p in ["fiebre", "cansancio", "malestar", "debilidad"]):
        return "médico general"

    return None

# =========================
# URGENCIA INTELIGENTE
# =========================

def detectar_urgencia(texto):
    texto = texto.lower()

    if any(p in texto for p in ["fuerte", "intenso", "grave", "no puedo"]):
        return "alta"

    if any(p in texto for p in ["dolor", "fiebre", "mareo"]):
        return "media"

    return "baja"

# =========================
# API
# =========================

@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.json

    if not data or "texto" not in data:
        return jsonify({"error": "No se envió texto"}), 400

    texto = data["texto"]

    # 1️⃣ reglas
    especialista = detectar_por_reglas(texto)

    # 2️⃣ IA si no encuentra
    if especialista is None:
        X_test = vectorizer.transform([texto])
        especialista = modelo.predict(X_test)[0]

    urgencia = detectar_urgencia(texto)

    return jsonify({
        "especialista": especialista,
        "urgencia": urgencia,
        "recomendacion": "Se recomienda acudir a consulta médica"
    })

if __name__ == "__main__":
    app.run(port=5000)