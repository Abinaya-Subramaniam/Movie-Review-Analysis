from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        
        prediction = model.predict([review])
        
        sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
        
        return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
