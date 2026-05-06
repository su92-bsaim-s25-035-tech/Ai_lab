from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv('amazon.csv', nrows=5000)
frequent_val = data['rating'].mode()[0]
data['rating'] = data['rating'].fillna(frequent_val)

for col in data.select_dtypes(include='object').columns:
    data[col] = pd.factorize(data[col])[0]

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

model = RandomForestClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form['features'].split()]
    prediction = model.predict([features])
    return render_template('index.html', prediction_text=f'Predicted Rating: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)