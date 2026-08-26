import pandas as pd
import json
import spacy
import re

# 1. CARGAR EL MODELO DE SPACY EN ESPAÑOL
try:
    nlp = spacy.load("es_core_news_sm")
    print("Modelo SpaCy (Español) cargado correctamente.")
except Exception as e:
    print("Error cargando SpaCy. Asegúrate de haber instalado el modelo de español ejecutando:")
    print("python -m spacy download es_core_news_sm")

# ==========================================
# SECCIÓN A: EXPLORACIÓN DE DATOS (EDA)
# ==========================================
def explorar_datasets():
    print("\n--- Explorando Dataset de IMDb (CSV) ---")
    try:
        df_imdb = pd.read_csv("dataset/IMDB Dataset SPANISH.csv")
        print(f"IMDb cargado. Total de filas: {len(df_imdb)}")
        print("Columnas disponibles:", df_imdb.columns.tolist())
        print(df_imdb.head(3))
    except Exception as e:
        print("Error cargando IMDb:", e)

    print("\n--- Explorando Dataset de Amazon (JSONL) ---")
    try:
        # Cargamos el archivo de entrenamiento que descargaste
        df_amazon_train = pd.read_json("dataset/train.jsonl", lines=True) 
        print(f"Amazon Train cargado. Total de filas: {len(df_amazon_train)}")
        print("Columnas disponibles:", df_amazon_train.columns.tolist())
        print(df_amazon_train.head(3))
    except Exception as e:
        print("Error cargando Amazon Train:", e)
        
    return df_imdb, df_amazon_train 


# ==========================================
# SECCIÓN B: PIPELINE DE LIMPIEZA NLP
# ==========================================
def limpiar_texto(texto):
    """
    Recibe un texto crudo y aplica el pipeline de NLP:
    - Minúsculas
    - Remoción de puntuación y números
    - Lematización (SpaCy)
    - Remoción de Stopwords (SpaCy)
    """
    if not isinstance(texto, str):
        return ""
        
    # 1. Minúsculas y limpieza básica (dejar solo letras)
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto) 
    
    # 2. Procesamiento con SpaCy
    doc = nlp(texto)
    
    tokens_limpios = []
    for token in doc:
        if not token.is_stop and token.text.strip():
            tokens_limpios.append(token.lemma_) # Extrae la raíz (lema)
            
    return " ".join(tokens_limpios)

if __name__ == "__main__":
    # 1. Exploramos las columnas de los archivos descargados
    df_imdb, df_amazon = explorar_datasets()
    
    # 2. Probamos que la función de limpieza esté trabajando correctamente
    texto_prueba = "¡Estos zapatos son pésimos! Los compré ayer y ya se están rompiendo. 100% no los recomendaría."
    print("\n--- Prueba de Limpieza NLP ---")
    print(f"Texto Original: {texto_prueba}")
    texto_limpio = limpiar_texto(texto_prueba)
    print(f"Texto Procesado (Lemas): {texto_limpio}")
