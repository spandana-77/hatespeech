import os
from flask import Flask, request, render_template
import joblib

# Configure Flask to search for templates and static files in the current folder (flat structure)
app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# Load the trained model pipeline
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentiment_model.pkl')
pipeline = joblib.load(model_path)

# Store last 5 predictions
history = []

@app.route('/')
def home():
    return render_template('index.html', history=history)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip()
    if not text:
        return render_template('index.html', 
                               result="⚠️ Empty Input", 
                               confidence=0.0, 
                               color="orange", 
                               text=text, 
                               history=history)

    # Perform prediction
    prediction = pipeline.predict([text])[0]
    confidence = pipeline.predict_proba([text])[0]
    confidence_score = round(max(confidence) * 100, 1)

    if prediction == 1:
        result = "🟢 Clean / Positive"
        color = "green"
    else:
        result = "🔴 Hate Speech / Negative"
        color = "red"

    # Add to history (keep last 5 only)
    history.insert(0, {
        'text': text,
        'result': result,
        'confidence': confidence_score,
        'color': color
    })
    
    if len(history) > 5:
        history.pop()

    return render_template('index.html',
                           result=result,
                           confidence=confidence_score,
                           color=color,
                           text=text,
                           history=history)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)