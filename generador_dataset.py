import pandas as pd
import random

def generar_dataset_tesis():
    print("Iniciando generador de corpus lingüístico...")
    dataset = []
    
    # 1. GENERADOR: RESEÑAS REALES (Label: 0) - 4000 reseñas
    sujetos_real = ["La habitación", "El hotel", "El lugar", "El cuarto", "El alojamiento"]
    detalles_real = [
        "estaba limpia pero el baño era muy pequeño.", 
        "tenía buena vista, aunque el aire acondicionado hacía ruido.",
        "está muy bien ubicado, a dos cuadras del metro.",
        "es cómodo, pero la cama era un poco dura para mi gusto.",
        "cumple con lo prometido, nada del otro mundo pero está bien para dormir."
    ]
    extras_real = [
        " El desayuno es normal, café y pan.", 
        " El personal de recepción tardó en darnos la llave.",
        " Había un poco de ruido de la calle en la noche.",
        " El wifi funcionaba bien en el lobby pero no arriba.",
        " Me gustó la piscina aunque cerraba muy temprano."
    ]
    
    for _ in range(4000):
        texto = f"{random.choice(sujetos_real)} {random.choice(detalles_real)}{random.choice(extras_real)}"
        dataset.append({"text": texto, "label": 0, "tipo": "Real"})

    # 2. GENERADOR: FRAUDE HUMANO (Estilo Ott) (Label: 1) - 2000 reseñas
    sujetos_human = ["¡Fue el mejor viaje de mi vida!", "¡Increíble!", "No puedo creer lo perfecto que fue todo.", "Mi esposo y yo quedamos fascinados.", "¡Wow! Simplemente espectacular."]
    detalles_human = [
        " Todo era lujo y perfección absoluta.",
        " Nos trataron como reyes desde que entramos.",
        " Es el lugar más hermoso en el que he estado jamás.",
        " Nunca había experimentado un servicio de tanta categoría.",
        " Todo fue un sueño hecho realidad, mágico."
    ]
    extras_human = [
        " Lo recomiendo al 1000% a todos mis amigos.",
        " Definitivamente volveremos todos los años.",
        " No busquen más, este es el lugar definitivo.",
        " Vale cada centavo, la mejor decisión de nuestras vidas.",
        " ¡Me sentí como en un palacio de cinco estrellas!"
    ]
    
    for _ in range(2000):
        texto = f"{random.choice(sujetos_human)}{random.choice(detalles_human)}{random.choice(extras_human)}"
        dataset.append({"text": texto, "label": 1, "tipo": "Fake_Human"})

    # 3. GENERADOR: FRAUDE I.A. (Label: 1) - 2000 reseñas
    sujetos_ai = ["El establecimiento proporciona", "Este hotel ofrece", "El alojamiento garantiza", "La propiedad presenta", "Este recinto destaca por"]
    detalles_ai = [
        " una experiencia inigualable para el huésped exigente.",
        " un nivel de confort y sofisticación excepcional.",
        " instalaciones inmaculadas y un servicio de primer nivel.",
        " una estadía que combina elegancia moderna con atención prístina.",
        " amenidades de clase mundial y una ubicación estratégica envidiable."
    ]
    extras_ai = [
        " Es importante destacar la eficiencia de su personal. En conclusión, es una opción altamente recomendable.",
        " Además, su oferta gastronómica es sobresaliente. En resumen, superará todas las expectativas.",
        " Adicionalmente, cuenta con espacios óptimos para el descanso. En definitiva, una elección superlativa.",
        " Cabe mencionar el impecable estado de las habitaciones. En síntesis, un servicio cinco estrellas indiscutible.",
        " Por consiguiente, representa la mejor relación calidad-precio del sector. Una experiencia verdaderamente memorable."
    ]
    
    for _ in range(2000):
        texto = f"{random.choice(sujetos_ai)}{random.choice(detalles_ai)}{random.choice(extras_ai)}"
        dataset.append({"text": texto, "label": 1, "tipo": "Fake_AI"})
        
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    ruta_guardado = "dataset/corpus_tesis_balanceado.csv"
    df.to_csv(ruta_guardado, index=False, encoding='utf-8')
    
    print(f"\n¡Éxito! Dataset sintético de {len(df)} reseñas creado en: {ruta_guardado}")
    print("\nDistribución de clases:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    generar_dataset_tesis()
