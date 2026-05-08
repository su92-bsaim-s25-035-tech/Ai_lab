from flask import Flask,render_template,request
import pickle 
app = Flask(__name__)

# Loading saved model
model= pickle.load(open('models/dress_code_model.pkl','rb'))
 
# Loading saved mappings
mappings =pickle.load(open('models/factorize_mappings.pkl','rb'))

# Home Page
@app.route('/')
def index():
    return render_template('index.html')
 
# Predict Page
@app.route('/predict', methods=['POST'])
def predict():
    # Getting data from form
    department = request.form['department']
    gender = request.form['gender']
    shirt_color = request.form['shirt_color']
    shirt_type = request.form['shirt_type']
    pants_type = request.form['pants_type']
    shoe_type = request.form['shoe_type']
    has_id_card = request.form['has_id_card']
    has_safety_gear = request.form['has_safety_gear']
 
    # Converting text to numbers using saved mappings
    data = [
        mappings['department'].get(department, 0),
        mappings['gender'].get(gender, 0),
        mappings['shirt_color'].get(shirt_color, 0),
        mappings['shirt_type'].get(shirt_type, 0),
        mappings['pants_type'].get(pants_type, 0),
        mappings['shoe_type'].get(shoe_type, 0),
        mappings['has_id_card'].get(has_id_card, 0),
        mappings['has_safety_gear'].get(has_safety_gear, 0),
    ]
 
    # Predicting using model
    result = model.predict([data])[0]
 
    # Sending result to result page
    return render_template('result.html', result=result, department=department)
 
# Rules Page
@app.route('/rules')
def rules():
    return render_template('rules.html')
 
# Running the app
if __name__ == '__main__':
    app.run(debug=True)
 