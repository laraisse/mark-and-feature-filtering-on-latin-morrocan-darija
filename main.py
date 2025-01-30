import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.metrics import classification_report
from transformers import pipeline
from flask import Flask, render_template, request, redirect, url_for
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict
from Levenshtein import distance as levenshtein_distance
from difflib import SequenceMatcher


# for darija, i will start by making it for english, then either i will take the darija text and translate it to english (most likely)
# or search for a darija database + the need of message cleaning (darija stop words) (very hard)

file_path = 'data_train.csv'
# Index(['Id', 'Tweet', 'following', 'followers', 'actions', 'is_retweet','location', 'Type'],dtype='object')
df = pd.read_csv(file_path)

df2 = df[['Type', 'Tweet']]
df2['type_value'] = df2['Type'].apply(lambda x: 1 if x == 'Spam' else 0)  # creating a row for numerical value (spam =1)

# for data cleaning we need to take out the spams : for that we do a double filter systeme black list +naive bayes modele
# first wash (black list user and black list message)
#https://huggingface.co/lachkarsalim/LatinDarija_Ehttps://huggingface.co/lachkarsalim/LatinDarija_English-v1nglish-v1
def translation(messages):
    pipe = pipeline("translation", model="lachkarsalim/LatinDarija_English-v1")
    m=[]
    for message in messages:
        m.append(pipe(message)[0]['translation_text'])
    return m
def message_spam(message): #black list of spam messages
    spam_message = open('spam_message.txt', 'r')
    spams = spam_message.readlines()
    spam_message.close()
    if message+'\n' in spams:
        return True
    else:
        return False


# second wash, AI filter (recognise is the message is spam or not )
def message_cleaner(message):  # will filter all stop words (like : you , are ...)
    filtered_tokens = [token for token in nltk.word_tokenize(message) if
                       token.lower() not in nltk.corpus.stopwords.words('english')]
    return [' '.join(filtered_tokens)]


def naive_bayes_filter(messages):  # naive bayses modele entry type == list
    predict =[]
    X_train, X_test, y_train, y_test = train_test_split(df2.Tweet, df2.type_value, test_size=0.2, random_state=42)
    cv = CountVectorizer()
    X_train_count = cv.fit_transform(X_train.values)
    X_test_count = cv.transform(X_test.values)
    mnb = MultinomialNB()
    mnb.fit(X_train_count, y_train)
    y_pred = mnb.predict(X_test_count)
    #print(mnb.score(X_test_count,y_test))  # prediction score
    #print(classification_report(y_test,y_pred))
    for message in messages:
        message_count = cv.transform(message_cleaner(message))
        predict.append(mnb.predict(message_count))
    return predict

def spam_detector(messages): #messages_type is list
    darija = messages
    tr_message = translation(messages)
    spam_message = open('spam_message.txt','a')
    non_spam_message = open('non_spam_messages.txt','a')
    prediction = naive_bayes_filter(tr_message)
    for i in range(len(tr_message)):
        if prediction[int(i)][0] == 1:
            spam_message.write(darija[i])
        else :
            non_spam_message.write(darija[i])
    spam_message.close()
    non_spam_message.close()

def levenshtein_similarity(word1: str, word2: str) -> float:
  """
  Calcule la similarité basée sur la distance de Levenshtein
  Adaptée pour les fautes de frappe et substitutions
  """
  word1, word2 = word1.lower(), word2.lower()
  max_len = max(len(word1), len(word2))
  if max_len == 0:
    return 0
  distance = levenshtein_distance(word1, word2)
  return 1 - (distance / max_len)


def sequence_similarity(word1: str, word2: str) -> float:
  """
  Calcule la similarité de séquence
  Meilleure pour détecter les parties communes
  """
  return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()


def phonetic_similarity(word1: str, word2: str) -> float:
  """
  Calcule une similarité phonétique simple
  Utile pour les erreurs phonétiques courantes
  """
  # Dictionnaire de remplacement pour les sons similaires
  replacements = {
    'a': 'a', 'e': 'a', 'é': 'a', 'è': 'a', 'ê': 'a',
    'i': 'i', 'y': 'i',
    'o': 'o', 'u': 'o',
    'k': 'q', 'c': 'q',
    'z': 's',
    'f': 'v',
    'b': 'p',
    't': 'd',
    'n': 'm'
  }

  # Simplifie les mots en remplaçant les caractères similaires
  def simplify(word: str) -> str:
    return ''.join(replacements.get(c, c) for c in word.lower())

  simple1 = simplify(word1)
  simple2 = simplify(word2)
  return sequence_similarity(simple1, simple2)


def calculate_brand_similarity(word: str, brand: str) -> Dict:
  """
  Calcule tous les scores de similarité entre un mot et une marque
  """
  lev_score = levenshtein_similarity(word, brand)
  seq_score = sequence_similarity(word, brand)
  phon_score = phonetic_similarity(word, brand)

  # Score composite (moyenne des trois scores)
  composite_score = (lev_score + seq_score + phon_score) / 3

  return {
    'found_word': word,
    'matched_brand': brand,
    'composite_score': round(composite_score, 3),
    'details': {
      'levenshtein_score': round(lev_score, 3),
      'sequence_score': round(seq_score, 3),
      'phonetic_score': round(phon_score, 3)
    }
  }
def find_similar_brands(text: str, brands: List[str], threshold: float = 0.7) -> List[Dict]:
  """
  Trouve toutes les marques similaires dans un texte

  Args:
      text: Texte à analyser
      brands: Liste des marques correctes
      threshold: Seuil minimum de similarité (0-1)
  """
  words = text.lower().split()
  matches = []

  for word in words:
    for brand in brands:
      similarity = calculate_brand_similarity(word, brand)
      if similarity['composite_score'] >= threshold:
        matches.append(similarity)

  return sorted(matches, key=lambda x: x['composite_score'], reverse=True)

def filterbrand(l,brands):
  v=[]
  nv=[]
  for i in l:
    matches = find_similar_brands(i[0], [brands], threshold=0.7)
    l1=[]
    for match in matches:
      l1.append(match['matched_brand'])
    if brands in l1:
      v.append(i)
    else:
      nv.append(i)
  return v,nv


def filterproduit(l,brands,product):
  v=[]
  nv=[]
  for i in l:
    matches = find_similar_brands(i, [brands], threshold=0.7)
    l1=[]
    for match in matches:
      l1.append(match['matched_brand'])
    if brands in l1:
      v.append(i)
    else:
       matches = find_similar_brands(i, product, threshold=0.7)
       l2=[]
       for match in matches:
          l2.append(match['matched_brand'])
       if l2!=[]:
          v.append(i)
       else:
          nv.append(i)
  return v,nv

prix=['taman','flous','derham','dh','dhs','rial','dirham','ghali','rghis','lprix','lpri','prix','L3aqa','L7ba','dollar','euro']
qualité = ["qualité, quality, kalité","jawda","ljawda","lqualité","lqualiti","kaliti"]

def price_quality_product(l,Matrix):
  p = []
  R= []
  for i in l:
    matches = find_similar_brands(i, Matrix, threshold=0.7)
    l1=[]
    for match in matches:
      l1.append(match['matched_brand'])
    if l1 != []:
        p.append(i)
    else:
        R.append(i)
  return p, R

def send_email (email,name):
    smtp_address = "smtp.gmail.com"
    smtp_port = 587  # For starttls
    # on rentre les informations sur notre adresse e-mail
    email_address = "laraisse66@gmail.com"
    email_password = 'zjql htgw exet fzjc '

    # on rentre les informations sur le destinataire
    email_receiver = email
    body = f'Bonjour Mr/Mme {name}\n'f'veuillez trouver ci-joint le fichier de votre donné filtrer\n'f'Cordialement'
    message = MIMEMultipart()
    message['From'] = email_address
    message['To'] = email_receiver
    message['Subject'] = 'data filtrée'

    message.attach(MIMEText(body, "plain"))

    nom_fichier = "message general.txt"
    le_fichier = open("message_géneral.txt", "rb")
    filename2 = "message sur qualite.txt"
    attachment2 = open("message_sur_qualité.txt", "rb")
    filename3 = "message sur prix.txt"
    attachment3 = open("message_sur_prix.txt", "rb")

    file = MIMEBase('application', 'octet-stream')
    file.set_payload((le_fichier).read())
    encoders.encode_base64(file)
    file.add_header('Content-Disposition', "attachment; filename= %s" % nom_fichier)
    message.attach(file)
    file3 = MIMEBase('application', 'octet-stream')
    file3.set_payload((attachment2).read())
    file3.add_header('Content-Disposition', "attachment; filename= %s" % filename2)
    message.attach(file3)
    file2 = MIMEBase('application', 'octet-stream')
    file2.set_payload((attachment3).read())
    file2.add_header('Content-Disposition', "attachment; filename= %s" % filename3)
    message.attach(file2)

    le_fichier.close()
    attachment3.close()
    attachment2.close()

    mail_server = smtplib.SMTP("smtp.gmail.com", smtp_port)
    mail_server.starttls()
    mail_server.login(email_address, email_password)
    mail_server.send_message(message)
    mail_server.quit()

app = Flask(__name__)


# Configurer le répertoire d'upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Route pour afficher le formulaire
@app.route("/")
def form():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Route pour traiter les données et le fichier uploadé
@app.route("/submit", methods=["POST"])
def submit():
    name = request.form["name"]
    email = request.form["email"]
    uploaded_file = request.files["file"]
    product_name = request.form["product name"]
    company_name = request.form["company name"]
    product_features = request.form["product features"]

    product_name = product_name.split(',')
    product_features = product_features.split()

    # Vérifier si un fichier a été uploadé
    if uploaded_file and uploaded_file.filename != "":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        return "Aucun fichier sélectionné !", 400

    with open(f'uploads/{uploaded_file.filename}', "r") as data:
        messages = data.readlines()
        v,nv = filterproduit(messages,company_name,product_name)
        data.close()
    spam_detector(v)
    with open('non_spam_messages.txt','r') as dt :
        pri = open('message_sur_prix.txt','w')
        qual =open('message_sur_qualité.txt','w')
        gen = open('message_géneral.txt','w')
        V = dt.readlines()
        p,R = price_quality_product(V,prix)
        q,S =price_quality_product(V,qualité)
        pri.writelines(p)
        qual.writelines(q)
        gen.writelines(R)
        gen.writelines(S)
        pri.close()
        qual.close()
        gen.close()
        dt.close()
    open('non_spam_messages.txt','w')


    send_email(email,name)

    # Stocker les informations dans un fichier texte
    with open("data.txt", "a") as file:
        file.write(f"Nom: {name}, Email: {email}, Fichier: {uploaded_file.filename}, product name: {product_name}, company name: {company_name}, product_features:{product_features}\n")

    return redirect(url_for("dashboard"))

@app.route("/compare", methods=["POST"])
def compare():
    name = request.form["name"]
    email = request.form["email"]
    uploaded_file = request.files["file"]
    product_name = request.form["product name"]
    company_name = request.form["company name"]
    product_features = request.form["product features"]

    product_name2 = request.form["product name2"]
    product_features2 = request.form["product features2"]

    product_name = product_name.split(',')
    product_name2 = product_name2.split(',')
    product_features = product_features.split()
    product_features2 = product_features2.split()

    # Vérifier si un fichier a été uploadé
    if uploaded_file and uploaded_file.filename != "":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
        uploaded_file.save(file_path)
    else:
        return "Aucun fichier sélectionné !", 400

    with open(f'uploads/{uploaded_file.filename}', "r") as data:
        messages = data.readlines()
        v,nv = filterproduit(messages, company_name, product_name)
        f,nf = filterproduit(messages,company_name,product_name2)
        data.close()
    spam_detector(v)
    spam_detector(f)
    with open('non_spam_messages.txt', 'r') as dt:
        pri = open('message_sur_prix.txt', 'w')
        qual = open('message_sur_qualité.txt', 'w')
        gen = open('message_géneral.txt', 'w')
        V = dt.readlines()
        p, R = price_quality_product(V, prix)
        q, S = price_quality_product(V, qualité)
        pri.writelines(p)
        qual.writelines(q)
        gen.writelines(R)
        gen.writelines(S)
        pri.close()
        qual.close()
        gen.close()
        dt.close()
    open('non_spam_messages.txt', 'w')

    send_email(email, name)

    # Stocker les informations dans un fichier texte
    with open("data.txt", "a") as file:
        file.write(
            f"Nom: {name}, Email: {email}, Fichier: {uploaded_file.filename}, product name: {product_name}, company name: {company_name}, product_features:{product_features}\n")

    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
