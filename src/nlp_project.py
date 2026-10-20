"""Reusable, leakage-aware SMS spam classification workflow."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, ConfusionMatrixDisplay, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/sms+spam+collection.zip"
REPORTS = ROOT / "reports"
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
SEED = 42
UCI_SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_corpus(archive: Path = ARCHIVE) -> pd.DataFrame:
    """Read the official headerless TSV from the UCI ZIP, without extraction."""
    with zipfile.ZipFile(archive) as zf:
        name = next(n for n in zf.namelist() if n.endswith("SMSSpamCollection"))
        with zf.open(name) as stream:
            df = pd.read_csv(stream, sep="\t", header=None, names=[TARGET_COLUMN, TEXT_COLUMN],
                             encoding="utf-8", dtype=str)
    validate_data(df)
    return df


def normalize_text(value: str) -> str:
    """Canonical key for exact/whitespace/case duplicates; model sees original text."""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def validate_data(df: pd.DataFrame) -> None:
    missing = {TEXT_COLUMN, TARGET_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if df[[TEXT_COLUMN, TARGET_COLUMN]].isna().any().any():
        raise ValueError("Missing values found in required columns")
    if df[TEXT_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Empty messages found")
    if set(df[TARGET_COLUMN].unique()) != {"ham", "spam"}:
        raise ValueError("Expected labels ham and spam")


def audit_data(df: pd.DataFrame, archive: Path = ARCHIVE) -> dict:
    keys = df[TEXT_COLUMN].map(normalize_text)
    label_counts_by_key = df.assign(_key=keys).groupby("_key")[TARGET_COLUMN].nunique()
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "sha256_zip": sha256(archive),
        "missing_required": int(df[[TEXT_COLUMN, TARGET_COLUMN]].isna().sum().sum()),
        "empty_text": int(df[TEXT_COLUMN].astype(str).str.strip().eq("").sum()),
        "duplicate_rows_exact": int(df.duplicated(subset=[TEXT_COLUMN]).sum()),
        "duplicate_rows_normalized": int(keys.duplicated().sum()),
        "conflicting_duplicate_groups": int((label_counts_by_key > 1).sum()),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "message_length_median": float(df[TEXT_COLUMN].str.len().median()),
    }


def clean_for_split(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove conflicting groups and duplicate normalized texts before splitting."""
    validate_data(df)
    work = df.copy()
    work["_key"] = work[TEXT_COLUMN].map(normalize_text)
    label_counts = work.groupby("_key")[TARGET_COLUMN].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    conflicts_removed = int(work["_key"].isin(conflicting_keys).sum())
    work = work.loc[~work["_key"].isin(conflicting_keys)].copy()
    duplicates_removed = int(work["_key"].duplicated().sum())
    work = work.drop_duplicates("_key", keep="first").drop(columns="_key").reset_index(drop=True)
    validate_data(work)
    return work, {"conflicting_rows_removed": conflicts_removed,
                  "duplicate_rows_removed": duplicates_removed}


def build_models() -> dict[str, Pipeline]:
    def tfidf() -> TfidfVectorizer:
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                               strip_accents="unicode", dtype=np.float64)
    return {
        "dummy": Pipeline([("tfidf", tfidf()),
                           ("model", DummyClassifier(strategy="most_frequent"))]),
        "complement_nb": Pipeline([("tfidf", tfidf()),
                                   ("model", ComplementNB(alpha=0.5))]),
        "logistic_regression": Pipeline([("tfidf", tfidf()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=SEED))]),
    }


def run_experiment(df: pd.DataFrame, reports_dir: Path = REPORTS) -> dict:
    """Fit comparable models and write all tabular and visual evidence."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    clean, cleaning = clean_for_split(df)
    x_train, x_test, y_train, y_test = train_test_split(
        clean[TEXT_COLUMN], clean[TARGET_COLUMN], test_size=0.25,
        random_state=SEED, stratify=clean[TARGET_COLUMN])
    metrics, fitted, predictions = [], {}, {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fitted[name] = model
        predictions[name] = pred
        metrics.append({"model": name,
                        "f1_macro": f1_score(y_test, pred, average="macro"),
                        "accuracy": accuracy_score(y_test, pred)})
    scores = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    selected = str(scores.iloc[0]["model"])
    selected_pred = predictions[selected]
    scores.to_csv(reports_dir / "text_metrics.csv", index=False, float_format="%.6f")
    report = classification_report(y_test, selected_pred, labels=["ham", "spam"],
                                   output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(reports_dir / "classification_report.csv", float_format="%.6f")
    prf = precision_recall_fscore_support(y_test, selected_pred, labels=["ham", "spam"],
                                         zero_division=0)
    class_metrics = pd.DataFrame({"label": ["ham", "spam"], "precision": prf[0],
        "recall": prf[1], "f1": prf[2], "support": prf[3]})
    class_metrics.to_csv(reports_dir / "class_metrics.csv", index=False, float_format="%.6f")
    audit = audit_data(df)
    with (reports_dir / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump({**audit, **cleaning, "clean_rows": len(clean), "train_rows": len(x_train),
                   "test_rows": len(x_test), "test_fraction": 0.25, "random_state": SEED},
                  stream, indent=2, ensure_ascii=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df[TARGET_COLUMN].value_counts().reindex(["ham", "spam"]).plot.bar(ax=axes[0], color=["#2563eb", "#f97316"])
    axes[0].set_title("Distribución de clases en corpus original")
    axes[0].set_ylabel("Mensajes")
    axes[0].set_xlabel("Clase")
    df.assign(longitud=df[TEXT_COLUMN].str.len()).boxplot(column="longitud", by=TARGET_COLUMN,
        ax=axes[1], grid=False, showfliers=False)
    axes[1].set_title("Longitud del mensaje por clase")
    axes[1].set_xlabel("Clase")
    axes[1].set_ylabel("Caracteres")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(reports_dir / "corpus_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    confusion = confusion_matrix(y_test, selected_pred, labels=["ham", "spam"])
    pd.DataFrame(confusion, index=["real_ham", "real_spam"],
                 columns=["pred_ham", "pred_spam"]).to_csv(reports_dir / "confusion_matrix.csv")
    ConfusionMatrixDisplay.from_predictions(y_test, selected_pred, labels=["ham", "spam"],
        display_labels=["Ham (legítimo)", "Spam"], cmap="Blues", values_format="d")
    plt.title(f"Matriz de confusión · {selected}")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()
    error_mask = (pd.Series(selected_pred, index=y_test.index) != y_test)
    errors = pd.DataFrame({"text": x_test[error_mask], "real": y_test[error_mask],
        "predicted": pd.Series(selected_pred, index=y_test.index)[error_mask]}).reset_index(drop=True)
    errors = errors.head(20).copy()
    annotations = [
        ("Yes.. now only saw", "Respuesta personal breve", "Pocas palabras y sin contexto; el mensaje no tiene señales promocionales claras."),
        ("Latest News!", "Titular humorístico", "La forma de titular y el juego de palabras pueden parecer una oferta; requiere contexto."),
        ("Are you free now?", "Ambigüedad léxica", "Free significa disponible en conversación personal, aunque también aparece en promociones."),
        ("I got a call from a landline", "Contexto ausente", "Relato personal con detalles locales; sin conversación previa el vocabulario es atípico."),
        ("dating:i have had", "Mención de campaña", "Habla de una campaña de radio y de un posible mensaje no deseado, no de una oferta propia."),
        ("Sorry! U can not unsubscribe", "Oferta y baja", "Lenguaje de suscripción y baja combinado con una respuesta de servicio puede confundir."),
        ("LookAtMe!:", "Contenido comercial adulto", "La compra y la llamada a la acción están expresadas de forma menos convencional."),
        ("Hi this is Amy, we will be sending", "Promoción conversacional", "El saludo personal precede a una invitación promocional de contenido adulto."),
        ("Babe: U want me", "Oferta adulta explícita", "Abreviaturas, tono conversacional y contenido adulto se combinan con una oferta SMS."),
        ("Your daily text from me", "Actualización personal breve", "Mensaje breve en primera persona; text puede coincidir con vocabulario promocional."),
        ("Sir, i am waiting for your call", "Solicitud personal breve", "Petición de devolución de llamada sin contexto previo."),
        ("Have you laid your airtel", "Referencia a servicio", "La mención de una marca telefónica puede activar patrones comerciales."),
        ("Dont you have message offer", "Oferta ambigua", "La frase es breve y puede describir una conversación o una promoción."),
        ("I'm at home. Please call", "Solicitud personal breve", "Pocas palabras y una petición genérica de llamada."),
        ("For sale - arsenal dartboard", "Anuncio clasificado", "Es una venta puntual de un artículo; la etiqueta spam puede depender del criterio del corpus."),
        ("Sorry I missed your call", "Número telefónico en contexto", "La presencia de un número y una conversación de llamada sin antecedentes."),
        ("Please protect yourself from e-threats", "Vocabulario de seguridad", "Términos como password y PIN aparecen en una advertencia legítima."),
        ("I'm at work. Please call", "Actualización personal breve", "Mensaje corto de estado personal, con escasa información contextual."),
        ("I'm vivek:)i got call", "Identidad y contexto ausentes", "El remitente intenta identificar una llamada previa que el modelo no observa."),
        ("ASKED 3MOBILE IF", "Queja y contexto ausente", "Texto de queja con operador, cantidades y abreviaturas; la etiqueta puede ser discutible sin procedencia."),
    ]
    categories, notes = [], []
    for text in errors[TEXT_COLUMN]:
        match = next(((category, note) for prefix, category, note in annotations if text.startswith(prefix)),
                     ("Revisar contexto", "Interpretación preliminar; verificar manualmente con el mensaje completo."))
        categories.append(match[0]); notes.append(match[1])
    errors["category_suggested"] = categories
    errors["analysis_suggested"] = notes
    errors["review_status"] = "Borrador asistido por IA; validar estudiante"
    errors.to_csv(reports_dir / "error_analysis.csv", index=False, encoding="utf-8-sig")
    return {"audit": audit, "cleaning": cleaning,
            "n_clean": int(len(clean)), "n_train": int(len(x_train)),
            "n_test": int(len(x_test)), "scores": scores, "selected": selected,
            "class_metrics": class_metrics, "confusion": confusion,
            "errors": errors, "fitted_models": fitted}


def main() -> dict:
    df = read_corpus()
    result = run_experiment(df)
    print("AUDITORÍA", result["audit"])
    print("LIMPIEZA", result["cleaning"])
    print(f"SPLIT train={result['n_train']} test={result['n_test']} seed={SEED}")
    print("MÉTRICAS\n", result["scores"].to_string(index=False))
    print("MÉTRICAS POR CLASE\n", result["class_metrics"].to_string(index=False))
    print("MATRIZ (filas=real ham,spam; columnas=predicción ham,spam)\n", result["confusion"])
    print("ERRORES DISPONIBLES", len(result["errors"]))
    print("Evidencia guardada en reports/")
    return result


if __name__ == "__main__":
    main()
