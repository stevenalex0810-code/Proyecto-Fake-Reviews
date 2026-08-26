import pandas as pd
import joblib
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
from nlp_pipeline import limpiar_texto # Importamos la limpieza que hicimos en la Fase 2

def entrenar_modelos():
    print("1. Cargando el dataset de Amazon...")
    df = pd.read_json("dataset/train.jsonl", lines=True)
    
    # pruebas iniciales, muestra de 5000 reseñas.
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    print(f"Total de reseñas a procesar (muestra): {len(df)}")
    
    print("2. Aplicando limpieza NLP (esto puede tardar unos minutos por la lematización)...")
    start_time = time.time()
    # Limpiamos el texto de la columna 'text'
    df['texto_limpio'] = df['text'].apply(limpiar_texto)
    print(f"Limpieza terminada en {round(time.time() - start_time, 2)} segundos.")
    
    # Asumimos que 'label' contiene la etiqueta numérica objetivo (0 o 1)
    X = df['texto_limpio']
    y = df['label'] 
    
    print("3. Vectorizando el texto (TF-IDF con unigramas y bigramas)...")
    vectorizador = TfidfVectorizer(ngram_range=(1, 2))
    X_vect = vectorizador.fit_transform(X)
    
    # Dividir en Entrenamiento (80%) y Prueba (20%) para las métricas finales
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
    

    # ENTRENAMIENTO: NAIVE BAYES
    print("\n--- ENTRENANDO NAIVE BAYES ---")
    nb_model = MultinomialNB()
    
    print("Aplicando Validación Cruzada 10-fold a Naive Bayes...")
    cv_scores_nb = cross_val_score(nb_model, X_train, y_train, cv=10, scoring='accuracy')
    print(f"Exactitud promedio (10-fold CV): {cv_scores_nb.mean():.4f}")
    
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    print("Resultados en conjunto de Prueba (20%):")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_nb, average='macro', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_nb, average='macro', zero_division=0):.4f}")
    
    # ENTRENAMIENTO: SVM (LinearSVC)
    print("\n--- ENTRENANDO SVM (LinearSVC) ---")
    # LinearSVC es una versión optimizada de SVM que corre mucho más rápido con texto
    svm_model = LinearSVC(random_state=42, dual=False)
    
    print("Aplicando Validación Cruzada 10-fold a SVM...")
    cv_scores_svm = cross_val_score(svm_model, X_train, y_train, cv=10, scoring='accuracy')
    print(f"Exactitud promedio (10-fold CV): {cv_scores_svm.mean():.4f}")
    
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print("Resultados en conjunto de Prueba (20%):")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_svm):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_svm, average='macro', zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred_svm, average='macro', zero_division=0):.4f}")
    
    # ==========================================
    # EXPORTACIÓN DEL MEJOR MODELO
    # ==========================================
    print("\n4. Exportando el modelo SVM (usualmente el más preciso en texto) y Vectorizador...")
    joblib.dump(svm_model, 'modelo_svm.pkl')
    joblib.dump(vectorizador, 'vectorizador.pkl')
    print("¡Archivos .pkl guardados exitosamente!")

if __name__ == "__main__":
    entrenar_modelos()
