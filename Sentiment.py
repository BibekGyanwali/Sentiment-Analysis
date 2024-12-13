from flask import Flask, request, jsonify
import joblib

# Load the model, vectorizer, and LabelEncoder
model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl") 

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']

        
        text_features = vectorizer.transform([text])

        
        prediction = model.predict(text_features)

        
        sentiment = label_encoder.inverse_transform(prediction)

        
        return jsonify({"Text": text, "Sentiment": sentiment[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)