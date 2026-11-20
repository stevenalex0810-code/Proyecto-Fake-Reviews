import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "modelo_svm.pkl"
VECTORIZER_PATH = PROJECT_DIR / "vectorizador.pkl"


@dataclass(frozen=True)
class ModelResources:
    model: Any
    vectorizer: Any


@dataclass(frozen=True)
class PredictionResult:
    label: int
    description: str


class ModelCompatibilityError(RuntimeError):
    pass


@st.cache_resource(show_spinner=False)
def load_resources() -> ModelResources:
    missing_files = [
        path.name for path in (MODEL_PATH, VECTORIZER_PATH) if not path.is_file()
    ]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"No se encontraron los archivos necesarios del modelo: {missing}."
        )

    try:
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning
    except ImportError as error:
        raise RuntimeError(
            "Faltan dependencias para cargar el modelo. Instálalas desde "
            "requirements.txt."
        ) from error

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
    except InconsistentVersionWarning as error:
        raise ModelCompatibilityError(
            "El modelo se creó con scikit-learn "
            f"{error.original_sklearn_version}, pero el entorno usa "
            f"{error.current_sklearn_version}. Instala Python 3.11 o posterior "
            "y las dependencias indicadas en requirements.txt."
        ) from error

    classes = set(model.classes_)
    if classes != {0, 1}:
        raise ValueError(
            "El modelo no contiene las etiquetas esperadas (0 = real, 1 = falsa)."
        )

    return ModelResources(model=model, vectorizer=vectorizer)


def validate_text(text: str) -> str:
    review = text.strip()
    if not review:
        raise ValueError("Escribe o pega una reseña antes de analizarla.")
    return review


def predict_review(text: str, resources: ModelResources) -> PredictionResult:
    from nlp_pipeline import limpiar_texto

    cleaned_text = limpiar_texto(text)
    transformed_text = resources.vectorizer.transform([cleaned_text])
    predicted_label = resources.model.predict(transformed_text)[0]

    if predicted_label == 0:
        return PredictionResult(label=0, description="Reseña clasificada como real")
    if predicted_label == 1:
        return PredictionResult(label=1, description="Reseña clasificada como falsa")

    raise ValueError(f"El modelo devolvió una etiqueta no reconocida: {predicted_label}")


def clear_form() -> None:
    st.session_state["review_text"] = ""
    st.session_state.pop("last_result", None)


def main() -> None:
    st.set_page_config(
        page_title="Fake Reviews Detector",
        page_icon="🔎",
        layout="centered",
    )
    st.markdown(
        """
        <style>
        .stApp { background-color: #f5f7fa; }
        .block-container { max-width: 900px; padding-top: 2.5rem; }
        .hero {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.5rem;
            border: 1px solid #d9e2ec;
            border-radius: 16px;
            background: linear-gradient(120deg, #ffffff, #edf3f8);
        }
        .hero p { color: #526273; margin-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Fake Reviews Detector</h1>
            <p>Analiza una reseña con el modelo SVM entrenado para identificar
            patrones asociados a reseñas reales o falsas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        resources = load_resources()
    except FileNotFoundError as error:
        logger.error("Faltan archivos del modelo: %s", error)
        st.error(str(error))
        st.stop()
    except ModelCompatibilityError as error:
        logger.error("%s", error)
        st.error(str(error))
        st.stop()
    except Exception:
        logger.exception("No fue posible cargar el modelo o el vectorizador.")
        st.error(
            "No se pudieron cargar los recursos del modelo. "
            "Revisa la instalación de dependencias y los archivos del proyecto."
        )
        st.stop()

    with st.form("review_form"):
        st.text_area(
            "Reseña del producto",
            placeholder="Escribe o pega aquí la reseña que deseas analizar...",
            height=220,
            key="review_text",
        )
        analyze_clicked = st.form_submit_button(
            "Analizar reseña",
            type="primary",
            use_container_width=True,
        )

    st.button(
        "Limpiar formulario",
        on_click=clear_form,
        use_container_width=True,
    )

    if analyze_clicked:
        st.session_state.pop("last_result", None)
        try:
            review = validate_text(st.session_state.get("review_text", ""))
        except ValueError as error:
            st.warning(str(error))
        else:
            try:
                result = predict_review(review, resources)
                st.session_state["last_result"] = {
                    "review": review,
                    "label": result.label,
                    "description": result.description,
                }
            except Exception:
                logger.exception("Error al procesar o clasificar la reseña.")
                st.error(
                    "No fue posible analizar la reseña. Verifica que el modelo "
                    "de español de SpaCy esté instalado y vuelve a intentarlo."
                )

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.subheader("Resultado del análisis")
        if result["label"] == 1:
            st.error(f"**{result['description']}**")
        else:
            st.success(f"**{result['description']}**")

        st.caption(
            "Esta respuesta es una predicción del modelo; no garantiza que la "
            "reseña sea auténtica o fraudulenta."
        )
        with st.container(border=True):
            st.markdown("**Reseña analizada**")
            st.write(result["review"])


if __name__ == "__main__":
    main()
