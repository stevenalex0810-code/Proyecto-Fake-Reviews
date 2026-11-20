from app import load_resources, predict_review, validate_text
from nlp_pipeline import limpiar_texto

recursos = load_resources()
print("Modelo cargado:", type(recursos.model).__name__)
print("Clases:", recursos.model.classes_.tolist())
print("Características TF-IDF:", recursos.model.n_features_in_)

resenas = [
    "La habitacion estaba limpia pero el bano era pequeno y el desayuno estuvo normal.",
    "Fue el mejor viaje de mi vida, todo fue perfecto y absolutamente maravilloso.",
]

for texto in resenas:
    texto = validate_text(texto)
    limpio = limpiar_texto(texto)
    matriz = recursos.vectorizer.transform([limpio])
    resultado = predict_review(texto, recursos)

    assert matriz.shape[0] == 1
    assert matriz.shape[1] == recursos.model.n_features_in_
    assert resultado.label in recursos.model.classes_

    print("\nTexto procesado:", limpio)
    print("Vector TF-IDF:", matriz.shape)
    print("Predicción:", resultado.label, "-", resultado.description)

try:
    validate_text("   ")
except ValueError:
    print("\nValidación de texto vacío: OK")
else:
    raise AssertionError("Se aceptó una reseña vacía")

print("\nBackend verificado correctamente.")