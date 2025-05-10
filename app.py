from flask import Flask, request, jsonify
import joblib

# Charger le modèle et le vectoriseur
model = joblib.load('logistic_regression_model.joblib')
vectorizer = joblib.load('vectorizer_reviews.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Lire du texte brut envoyé avec Content-Type: text/plain
    review = request.data.decode('utf-8')

    # Vectorisation
    review_vectorized = vectorizer.transform([review])

    # Prédiction
    prediction = model.predict(review_vectorized)
    result = "Positive" if prediction[0] == 1 else "Negative"

    # Réponse JSON
    return jsonify({
        "review": review,
        "prediction": result
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
