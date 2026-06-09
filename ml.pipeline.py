import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def generate_synthetic_data(num_samples=2000):
    """Generate a synthetic dataset of positive/clean reviews and negative/hate speech reviews."""
    positive_words = [
        "love", "great", "fantastic", "amazing", "beautiful", "superb", "brilliant", 
        "excellent", "delightful", "good", "best", "wonderful", "joy", "stunning",
        "kind", "helpful", "respectful", "friendly", "supportive", "polite", "happy",
        "nice", "awesome", "perfect", "glad", "admire", "pleasure", "pure"
    ]
    
    hate_negative_words = [
        "hate", "terrible", "boring", "awful", "poor", "dull", "disaster", "horrible", 
        "worst", "pain", "bad", "ugly", "depressing", "confusing", "disappointing",
        "stupid", "trash", "disgusting", "idiot", "garbage", "harass", "abuse", "loser", 
        "nasty", "toxic", "kill", "destroy", "dumb", "hate speech", "offensive", "vulgar"
    ]
    
    neutral_words = [
        "movie", "film", "acting", "plot", "story", "director", "visuals", "experience", 
        "performance", "watch", "character", "scene", "the", "a", "an", "and", "but", 
        "is", "was", "it", "they", "we", "he", "she", "this", "that", "there", "here",
        "product", "item", "service", "customer", "quality", "price", "speed", "delivery"
    ]

    np.random.seed(42)
    data = []
    
    for _ in range(num_samples):
        # Label 0: Hate Speech / Negative review
        # Label 1: Clean / Positive review
        label = np.random.choice([0, 1])
        
        if label == 1:
            # Pick positive words
            words = np.random.choice(positive_words, size=np.random.randint(2, 6)).tolist()
        else:
            # Pick negative/hate speech words
            words = np.random.choice(hate_negative_words, size=np.random.randint(2, 6)).tolist()
            
        # Add neutral words to construct a sentence
        words += np.random.choice(neutral_words, size=np.random.randint(3, 8)).tolist()
        np.random.shuffle(words)
        
        sentence = " ".join(words) + "."
        data.append({"text": sentence, "label": label})
        
    return pd.DataFrame(data)

def main():
    print("Generating synthetic sentiment & hate speech dataset...")
    df = generate_synthetic_data(2000)
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("Class distribution in training:")
    print(y_train.value_counts())
    
    # Define sklearn Pipeline
    print("Training scikit-learn Pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=3.0,
            max_iter=1000,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel trained successfully. Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Hate Speech / Negative', 'Clean / Positive']))
    
    # Save the pipeline to the current directory
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentiment_model.pkl')
    print(f"Saving model pipeline to {model_path}...")
    joblib.dump(pipeline, model_path)
    print("[SUCCESS] Model saved successfully!")

if __name__ == "__main__":
    main()