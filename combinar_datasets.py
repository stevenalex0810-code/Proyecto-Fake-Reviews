import pandas as pd
from deep_translator import GoogleTranslator
import time

def traducir_y_combinar():
    print("1. Cargando dataset original de Ott (deceptive-opinion.csv)...")
    try:
        # El dataset de Ott suele tener la columna 'deceptive' y 'text'
        df_ott = pd.read_csv("dataset/deceptive-opinion.csv")
    except FileNotFoundError:
        print("ERROR: No se encontró 'deceptive-opinion.csv' en la carpeta dataset.")
        print("Por favor, asegúrate de descargarlo y ponerlo en la ruta correcta.")
        return

    print("2. Iniciando traducción al español (esto tomará varios minutos)...")
    traductor = GoogleTranslator(source='en', target='es')
    textos_traducidos = []
    
    total_filas = len(df_ott)
    for index, fila in df_ott.iterrows():
        try:
            # Traduce el texto de la reseña
            texto_es = traductor.translate(fila['text'])
        except Exception as e:
            texto_es = fila['text'] # En caso de error de conexión, guarda el original
            
        textos_traducidos.append(texto_es)
        
        # Reporte de progreso
        if (index + 1) % 100 == 0:
            print(f"Traducidas {index + 1} de {total_filas} reseñas...")
            time.sleep(1) # Pausa para no saturar el servidor de traducción
            
    # Guardamos las traducciones en el DataFrame
    df_ott['texto_espanol'] = textos_traducidos
    
    # 3. Formateamos las etiquetas para que coincidan con tu modelo
    print("\n3. Estandarizando etiquetas para la Tesis...")
    # 'deceptive' en el dataset de Ott suele contener "truthful" o "deceptive"
    df_ott['label'] = df_ott['deceptive'].apply(lambda x: 1 if str(x).lower().strip() == 'deceptive' else 0)
    
    df_ott_limpio = pd.DataFrame({
        'text': df_ott['texto_espanol'],
        'label': df_ott['label'],
        'tipo': df_ott['deceptive'].apply(lambda x: "Fake_Human_Ott" if str(x).lower().strip() == 'deceptive' else "Real_Ott")
    })
    
    # 4. Fusionamos con el Dataset Sintético (Inteligencia Artificial)
    print("4. Cargando el dataset sintético que generamos con I.A...")
    df_sintetico = pd.read_csv("dataset/corpus_tesis_balanceado.csv")
    
    print("5. Fusionando ambos datasets...")
    df_final = pd.concat([df_sintetico, df_ott_limpio], ignore_index=True)
    
    # Mezclamos todas las filas aleatoriamente
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Exportamos el resultado final
    ruta_final = "dataset/dataset_tesis_definitivo.csv"
    df_final.to_csv(ruta_final, index=False, encoding='utf-8')
    
    print(f"\n¡Éxito! Dataset DEFINITIVO creado en: {ruta_final}")
    print("\nDistribución Total (0=Real, 1=Falsa):")
    print(df_final['label'].value_counts())
    print("\nTipos de reseñas incluidas en tu Tesis:")
    print(df_final['tipo'].value_counts())
    print("\n¡Ya puedes correr 'python modelos.py' cambiando la ruta de carga a 'dataset/dataset_tesis_definitivo.csv'!")

if __name__ == "__main__":
    traducir_y_combinar()
