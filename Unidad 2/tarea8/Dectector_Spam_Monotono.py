import pandas as pd
import re
from urllib.parse import urlparse
from datetime import datetime
from collections import Counter

#Se carga el dataset
try:
    df = pd.read_csv("U2/tarea8/spam_assassin.csv")
except FileNotFoundError:
    print("Error: El archivo spam_assassin.csv no se encontró.")
    exit()

#Creamos las funciones para la identificacion de caracteristicas de correo
def extraer_remitente(text):
    remitente_match = re.search(r"From: (.*?)(?:\n|$)", text, re.IGNORECASE)
    if remitente_match:
        return remitente_match.group(1).strip()
    return None

def extraer_asunto(text):
    asunto_match = re.search(r"Subject: (.*?)(?:\n|$)", text, re.IGNORECASE)
    if asunto_match:
        return asunto_match.group(1).strip()
    return None

def extraer_cuerpo(text):
    cuerpo_inicio = re.search(r"\n\s*\n", text)
    if cuerpo_inicio:
        return text[cuerpo_inicio.start():].strip()
    return text.strip() 

#Obtiene el remitente, asunto y cuerpo de cada elemento en el dataset
df['sender'] = df['text'].apply(extraer_remitente)
df['subject'] = df['text'].apply(extraer_asunto)
df['body'] = df['text'].apply(extraer_cuerpo)

#Reglas de clasificacion de spam
def reglas_de_spam(row): 
    texto = row.get('text', '')
    remitente = row.get('sender', '')
    asunto = row.get('subject', '')
    cuerpo = row.get('body', '')

    # Regla 0: Descartar "Re:" en el asunto
    if asunto and asunto.lower().startswith("re:"):
        return 0  # Marcar como no spam
    
    # Regla 1: Asunto con palabras clave sospechosas 
    palabras_spam_asunto = {"half price", "free", "winner", "urgent", "limited offer", "claim now", "lottery", "earn money", "click now", "prize", "congratulations", "guaranteed", "cash"}
    if asunto and any(word in asunto.lower() for word in palabras_spam_asunto):
        return 1

    # Regla 2: Cuerpo con palabras clave de spam 
    keywords_spam_cuerpo = { "to be removed from", "claim your prize", "limited time offer", "unsubscribe now", "click here", "congratulations", "fast money", "limited time", "buy now", "guaranteed", "cash prize", "special offer", "earn money online", "work from home", "investment opportunity"}
    if any(keyword in cuerpo.lower() for keyword in keywords_spam_cuerpo):
        return 1
    
    # Regla 3: Remitente con un dominio sospechoso o genérico
    if re.search(r"@(mail|web|promo|free)[0-9]+\.com$", remitente, re.IGNORECASE):
        return 1

    # Regla 4: Exceso de signos de exclamación (indicador de urgencia o spam)
    if cuerpo.count("!") > 10:
        return 1

    # Regla 5: Uso de frases en mayúsculas típicas de spam
    if re.search(r"(WIN|FREE|CLICK NOW|URGENT|LIMITED OFFER|BUY NOW|MONEY BACK GUARANTEE)", cuerpo):
        return 1

    # Regla 6: Exceso de etiquetas HTML sospechosas
    etiquetas_html_sospechosas = re.findall(r'<(font|table|td|tr|div|span|a|img|p|meta|input)>' , cuerpo , re.IGNORECASE)
    if len(etiquetas_html_sospechosas) > 0:
        return 1
    
    return 0

#Se aplican las reglas al dataset 
df['Spam'] = df.apply(reglas_de_spam, axis=1)

#Se evalua el rendimiento/precision
from sklearn.metrics import accuracy_score, classification_report

Precision = accuracy_score(df['target'], df['Spam'])
print(f"\nPrecisión del detector de spam basado en reglas: {Precision:.2f}")
print("\nReporte de Clasificación:\n", classification_report(df['target'], df['Spam']))

#Mostramos algunos correos evaluados y su clasificaciones 
print("\nPrimeros 50 correos electrónicos con sus predicciones:")
print(df[['subject', 'target', 'Spam']].head(50))

# Solicitar datos de un correo electrónico manualmente
print("\nIngrese los datos de un correo para analizar si es spam:")
remitente_manual = input("Remitente: ")
asunto_manual = input("Asunto: ")
cuerpo_manual = input("Cuerpo: ")

# Crear un diccionario con los datos ingresados
correo_manual = {
    'sender': remitente_manual,
    'subject': asunto_manual,
    'body': cuerpo_manual,
    'text': f"From: {remitente_manual}\nSubject: {asunto_manual}\n\n{cuerpo_manual}"
}

# Evaluar el correo ingresado
resultado_spam = reglas_de_spam(correo_manual)

if resultado_spam == 1:
    print("\nEl correo ingresado es SPAM.")
else:
    print("\nEl correo ingresado NO es spam.")
