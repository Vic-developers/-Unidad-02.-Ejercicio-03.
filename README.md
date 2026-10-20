# Ejercicio 03 - Corpus público y clasificador de texto reproducible

Proyecto reproducible de detección de SMS no deseados con el corpus SMS Spam Collection de UCI. La pregunta es: **¿cómo se compara TF-IDF con Complement Naive Bayes y regresión logística para distinguir SMS legítimos (ham) de spam en una partición sin textos duplicados entre entrenamiento y prueba?**

Autor académico: Victor Elvis Lorenzo  
Repositorio de entrega: <https://github.com/Vic-developers/-Unidad-02.-Ejercicio-03..>

## Resultados de esta ejecución

Se ejecutó la comparación con la descarga UCI incluida en `data/raw/`. La colección contiene 5,572 mensajes: 4,825 ham y 747 spam. Se encontraron 403 duplicados exactos y 12 duplicados adicionales al normalizar mayúsculas y espacios; no hubo etiquetas conflictivas. Para evitar fuga, las 415 repeticiones se eliminaron **antes** de separar los datos. La partición estratificada reservó 25% para prueba (semilla 42).

| Método | F1 macro | Exactitud |
|---|---:|---:|
| Regresión logística TF-IDF | 0.959727 | 0.982171 |
| Complement Naive Bayes TF-IDF | 0.953372 | 0.979845 |
| Baseline Dummy | 0.466722 | 0.875194 |

En prueba, la matriz del modelo seleccionado es `[[1115, 14], [9, 152]]`, con filas ham/spam reales y columnas ham/spam predichas. La regresión logística fue la mejor entre las tres en esta partición. Su recall para spam fue 0.944; 9 de 161 mensajes spam quedaron como ham. Esto no demuestra rendimiento en SMS actuales, español ni otros canales.

Las cifras se recalculan con `python scripts/run_analysis.py` o ejecutando la notebook. Los reportes visuales y tabulares están en `reports/`. El análisis de errores contiene 20 predicciones incorrectas reales del conjunto de prueba. Sus categorías y explicaciones son notas preliminares y requieren revisión del estudiante antes de defenderlas.

## Instalación y ejecución desde cero

Requiere Python 3.11 o 3.12 y una conexión a Internet para `pip install`. El ZIP de UCI se incluye para que el experimento no dependa de volver a descargar los datos.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
python scripts/run_analysis.py
jupyter notebook notebooks/ejercicio_03_sms.ipynb
```

En macOS/Linux, crea el entorno con `python3 -m venv .venv`, actívalo con `source .venv/bin/activate` y ejecuta los mismos comandos Python.

## Estructura

```text
data/raw/                 ZIP oficial UCI con su archivo descriptivo
docs/                     ficha del conjunto, fuentes y declaración de IA
notebooks/                notebook ejecutado con salidas guardadas
reports/                  métricas, matriz, auditoría, análisis de errores y gráficos
scripts/run_analysis.py   punto de entrada reproducible
src/nlp_project.py        carga, auditoría, limpieza, pipelines y evaluación
tests/                    pruebas del contrato de datos y la fuga por duplicados
requirements.txt          dependencias de ejecución y prueba
```

## Decisiones y límites

- Se conservan los mensajes originales en inglés; la normalización de espacios y mayúsculas solo se utiliza para detectar duplicados.
- Los modelos comparables comparten TF-IDF de palabras con unigramas y bigramas. Solo cambia el clasificador; Dummy fija una referencia de clase mayoritaria.
- Todo el vocabulario TF-IDF se aprende dentro de cada `Pipeline` al ajustar solo con entrenamiento.
- Complement Naive Bayes y regresión logística son pipelines de bolsa de palabras; no se les atribuye comprensión semántica.
- La partición es única y no mide cambio temporal ni rendimiento fuera de UCI. El baseline refleja el fuerte desbalance de clases.
- La tabla de errores es una muestra ordenada de hasta 20 errores del modelo seleccionado, no una estimación de la frecuencia poblacional de cada categoría.

## Datos, licencia y atribución

SMS Spam Collection, UCI Machine Learning Repository, DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84), licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Colección de mensajes SMS en inglés y etiquetas ham/spam; ver `docs/DATASET_CARD.md` y el archivo `readme` incluido en el ZIP. Atribución: Almeida, T. A., Gómez Hidalgo, J. M. y Yamakami, A. (2011), *Contributions to the Study of SMS Spam Filtering: New Collection and Results*, ACM Symposium on Document Engineering, DOI 10.1145/2034691.2034742.

Los mensajes se usan con fines académicos. La licencia del corpus no convierte el prototipo en un filtro apto para producción.

## Uso de IA

Se utilizó OpenAI Codex como apoyo para corregir errores de código, proponer pruebas y revisar la redacción. También se consultó para preparar notas preliminares sobre errores de clasificación; el estudiante debe revisarlas y explicar su interpretación. Se verificaron la descarga, la huella, los conteos, las métricas, la matriz de confusión y las pruebas automatizadas. El detalle y el prompt relevante están en `docs/AI_USAGE.md`.
