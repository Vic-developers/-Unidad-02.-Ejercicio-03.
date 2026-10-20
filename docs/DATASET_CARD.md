# Ficha del conjunto de datos

## Identificación

- Nombre: SMS Spam Collection
- Fuente: UCI Machine Learning Repository, dataset 228
- DOI: 10.24432/C5CC84
- URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Responsable de compilación: Tiago A. Almeida, José María Gómez Hidalgo y Akebo Yamakami
- Versión: archivo ZIP público incluido en `data/raw/sms+spam+collection.zip`
- SHA-256 del ZIP incluido: `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3`
- Licencia declarada por UCI: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Idioma predominante: inglés

## Propósito y variable objetivo

Clasificar cada mensaje como `ham` (mensaje legítimo) o `spam` (mensaje no deseado) para estudiar filtros de SMS. La pregunta del proyecto es cómo se comparan Complement Naive Bayes y regresión logística con TF-IDF frente a un baseline mayoritario, usando una partición de prueba independiente de textos repetidos.

## Diccionario

| Columna | Tipo | Descripción | Valores |
|---|---|---|---|
| `label` | categórica | Etiqueta de clase asignada al mensaje | `ham`, `spam` |
| `text` | texto | Cuerpo del SMS | Mensajes en inglés |

## Procedimiento de obtención

El ZIP oficial se obtuvo de UCI el 20 de octubre de 2026 y se conserva en `data/raw/`. `src/nlp_project.py` lee el archivo tabulado sin encabezado directamente desde el ZIP. La ejecución registra SHA-256. El archivo descargado no se modifica antes de auditarlo.

## Calidad observada en esta versión

- 5,572 filas; 0 nulos en texto/etiqueta y 0 textos vacíos.
- 4,825 ham y 747 spam.
- 403 filas duplicadas de forma exacta.
- 415 repeticiones al normalizar espacios y mayúsculas (incluye los duplicados exactos y 12 coincidencias adicionales).
- 0 grupos repetidos con etiquetas diferentes.
- Mediana de longitud: 62 caracteres.
- Tras retirar 415 repeticiones: 5,157 textos únicos. Partición estratificada 75/25, semilla 42: 3,867 entrenamiento y 1,290 prueba.

La huella se refiere al ZIP conservado, no a una copia externa del TSV. Los conteos de esta ficha se recalculan al ejecutar el pipeline.

## Población cubierta y excluida

Mensajes SMS en inglés reunidos de fuentes y periodos descritos por UCI, con clases ham/spam. No representa por diseño mensajes actuales, mensajería de aplicaciones, otros idiomas, regiones específicas ni el español dominicano.

## Riesgos, sesgos y usos prohibidos

La clase está desbalanceada y el corpus es histórico. Puede contener patrones de recolección y vocabulario que no se mantienen. No desplegar para bloquear mensajes reales, inferir intención, evaluar personas ni tomar decisiones sensibles sin datos representativos, pruebas temporales y revisión humana. Las predicciones no explican por qué se etiqueta un mensaje.

## Atribución

Almeida, T. A., Gómez Hidalgo, J. M. y Yamakami, A. (2011). “Contributions to the Study of SMS Spam Filtering: New Collection and Results.” ACM Symposium on Document Engineering. DOI: https://doi.org/10.1145/2034691.2034742. Corpus distribuido por UCI bajo CC BY 4.0.
