"""Execute notebook cells in order and save standard nbformat 4 JSON output."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "notebooks/ejercicio_03_sms.ipynb"

def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source}

cells = [
    markdown("""# Ejercicio 03 · Corpus público y clasificador de texto reproducible

**Autor:** Edwin Ramón José Nolasco  
**Pregunta:** ¿cómo se compara TF-IDF con Complement Naive Bayes y regresión logística para distinguir SMS legítimos (ham) de spam sin compartir textos duplicados entre entrenamiento y prueba?

Esta notebook carga el ZIP público de UCI, audita calidad y duplicados, separa antes de ajustar el vocabulario, compara tres modelos en el mismo holdout y registra errores. La descarga incluida permite repetirla sin red.

Corpus: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), licencia CC BY 4.0. Para instalar y ejecutar desde cero, consulta `../README.md`.
"""),
    code("""from pathlib import Path
import sys
ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from nlp_project import read_corpus, audit_data, clean_for_split, run_experiment
import pandas as pd
"""),
    markdown("## 1. Auditoría del corpus\n\nSe valida esquema, clases, ausencias y duplicados antes de entrenar."),
    code("""raw = read_corpus()
audit = audit_data(raw)
print(f"Dimensiones: {raw.shape[0]:,} filas × {raw.shape[1]} columnas")
print(f"SHA-256 ZIP: {audit['sha256_zip']}")
print(f"Etiquetas: {audit['class_counts']}")
print(f"Nulos / textos vacíos: {audit['missing_required']} / {audit['empty_text']}")
print(f"Duplicados exactos: {audit['duplicate_rows_exact']}")
print(f"Duplicados normalizados: {audit['duplicate_rows_normalized']}")
print(f"Grupos con etiquetas en conflicto: {audit['conflicting_duplicate_groups']}")
print(f"Mediana de caracteres: {audit['message_length_median']:.0f}")
"""),
    markdown("## 2. Prevención de fuga y evaluación\n\nSe excluyen grupos con etiquetas en conflicto y se conserva una instancia por texto normalizado. Luego se hace una partición estratificada 75/25, semilla 42. Los vectorizadores TF-IDF viven dentro de cada pipeline: el vocabulario se aprende solo con entrenamiento."),
    code("""clean, cleaning = clean_for_split(raw)
print(f"Filas retiradas por conflictos: {cleaning['conflicting_rows_removed']}")
print(f"Duplicados retirados antes del split: {cleaning['duplicate_rows_removed']}")
print(f"Textos únicos: {len(clean):,}")
result = run_experiment(raw)
print(f"Train: {result['n_train']:,}; prueba: {result['n_test']:,}; semilla: 42")
print(f"Modelo seleccionado por F1 macro: {result['selected']}")
print("\\nComparación (mismo split):")
print(result['scores'].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
"""),
    markdown("## 3. Métricas, clases y matriz\n\nLas métricas por clase exponen el rendimiento de spam frente al de ham. La matriz usa filas = clase real y columnas = clase predicha."),
    code("""print(result['class_metrics'].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("Matriz de confusión [ham, spam]:")
print(result['confusion'])
print(f"Errores del modelo seleccionado en el holdout: {len(result['errors'])}")
"""),
    markdown("""### Perfil del corpus

![Distribución de clases y longitudes](../reports/corpus_profile.png)

### Matriz de confusión

![Matriz de confusión](../reports/confusion_matrix.png)
"""),
    markdown("## 4. Revisión de 20 errores\n\nSon 20 predicciones incorrectas reales del holdout. Las categorías y notas son un borrador asistido por IA, no una validación del estudiante. Revise el contexto, corrija las etiquetas de análisis si hace falta y explique los patrones antes de entregar."),
    code("""errors = pd.read_csv(ROOT / "reports/error_analysis.csv")
print(errors[["text", "real", "predicted", "category_suggested", "analysis_suggested"]]
      .to_string(index=False, max_colwidth=100))
"""),
    markdown("## 5. Interpretación y límites\n\nEn esta ejecución, regresión logística TF-IDF obtuvo el F1 macro más alto (0.959727), por encima de Complement Naive Bayes (0.953372); el baseline mayoritario obtuvo 0.466722. El recall de spam fue 0.9441: 9 de 161 mensajes spam se clasificaron como ham. Es una comparación en una partición del corpus histórico en inglés, no una estimación para mensajes actuales o español dominicano. La ficha, tabla de errores, métricas y auditoría reproducible se guardan en `reports/` y `docs/`."""),
]

namespace = {"__name__": "__notebook__", "__file__": str(OUT)}
count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    count += 1
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(cell["source"], namespace)
        output = buffer.getvalue()
        if output:
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output}]
        cell["execution_count"] = count
    except Exception as exc:
        cell["outputs"] = [{"output_type": "error", "ename": type(exc).__name__,
                           "evalue": str(exc), "traceback": [repr(exc)]}]
        raise

OUT.parent.mkdir(parents=True, exist_ok=True)
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
    "language": "python", "name": "python3"}, "language_info": {"name": "python",
    "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}},
    "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Notebook ejecutada y guardada: {OUT}")
