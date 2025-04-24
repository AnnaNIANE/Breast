from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import joblib
from functools import wraps
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secure key in production

# Charge le modèle et le scaler
def load_model():
    model = joblib.load('models/breast_cancer_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, feature_names

# Crée la base de données et la table des utilisateurs si elles n'existent pas
def create_db():
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()

    # Créer une table pour stocker les utilisateurs
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL)''')

    connection.commit()
    connection.close()

# Appelle la fonction pour créer la base de données et la table
create_db()

# Fonction pour ajouter un utilisateur à la base de données
def add_user(username, password):
    hashed_password = generate_password_hash(password)
    
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()

    try:
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        connection.commit()
    except sqlite3.IntegrityError:
        print(f"Le nom d'utilisateur {username} existe déjà.")
    finally:
        connection.close()

# Fonction pour valider les informations de connexion
def validate_user(username, password):
    connection = sqlite3.connect('users.db')
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()

    connection.close()

    if user:
        return check_password_hash(user[2], password)  # user[2] est le mot de passe haché
    return False

# Charge le modèle, scaler et les noms des features
model, scaler, feature_names = load_model()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if validate_user(username, password):
            session['logged_in'] = True
            session['username'] = username
            flash('You are now logged in', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid credentials', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        password_confirm = request.form['password_confirm']

        if password != password_confirm:
            flash("Passwords do not match", 'danger')
        else:
            try:
                add_user(username, password)
                flash("User registered successfully", 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash("Username already exists", 'danger')

    return render_template('register.html')

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Obtenir les valeurs des caractéristiques depuis le formulaire
        features = []
        for feature in feature_names:
            value = request.form.get(feature, 0)
            features.append(float(value))
        
        # Normaliser les caractéristiques et faire la prédiction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][prediction]
        
        result = {
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'probability': f"{probability * 100:.2f}%"
        }
        
        return render_template('result.html', result=result)
    
    return render_template('predict.html', features=feature_names)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
