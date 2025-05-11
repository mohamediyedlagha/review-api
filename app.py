from flask import Flask, request, jsonify
import joblib

# Charger le modèle et le vectoriseur
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('vectorizer_reviews.joblib')

# Initialiser l'application Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Lire le champ "review" depuis multipart/form-data
    review = request.form.get('review', '')

    if not review:
        return jsonify({
            "error": "Missing 'review' field in multipart form data",
            "review": review,
            "prediction": None
        }), 400

    # Vectorisation
    review_vectorized = vectorizer.transform([review])

    # Prédiction
    prediction = model.predict(review_vectorized)
    result = "Positive" if prediction[0] == 1 else "Negative"

    # Retourner la réponse JSON
    return jsonify({
        "review": review,
        "prediction": result
    })

# Lancer le serveur (non utilisé sur Render, mais utile en local)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
