# Declaración de uso de IA

## Herramienta

OpenAI Codex, asistente de programación basado en GPT-6, usado durante la preparación de esta entrega.

## Prompt relevante

Prompt del usuario en este chat: “Indicame que necesitas para logar lo que me solicita esto y me entregues lo que solicita que cumpla con todo”. La consigna adjunta pidió “Integrar, de manera individual, las evidencias producidas en LAB04 y LAB05 en una entrega ejecutable y defendible” e indicó: “No se acepta una solución que no pueda ejecutarse desde las instrucciones del repositorio”. La consigna también enumeró notebook ejecutado, código reutilizable, pruebas, reportes, requirements.txt, README, enlace al repositorio y análisis de 20 errores.

## Partes asistidas

- Revisión de estructura y detección de diferencias entre el CSV demostrativo y los resultados previos descritos en el DOCX.
- Preparación del flujo de descarga/carga, auditoría, normalización de duplicados, pipelines, reportes, pruebas y notebook.
- Redacción de síntesis y propuestas de categorías y explicaciones para los errores del conjunto de prueba.

## Verificación realizada

- Se descargó el ZIP de UCI, se calculó SHA-256 y se comprobaron filas, columnas, nulos, clases y duplicados.
- Se ejecutó el pipeline con una partición estratificada fijada en semilla 42 y se generaron métricas y matriz de confusión.
- Se ejecutaron las pruebas automatizadas descritas en el README.
- La notebook guarda las salidas de su última ejecución.

## Correcciones y responsabilidad

Se reemplazaron los resultados del CSV demostrativo por una nueva ejecución del corpus UCI. La revisión encontró 415 duplicados normalizados y se retiraron antes de la separación para reducir fuga entre conjuntos. Se añadió el baseline y se compararon los dos pipelines bajo la misma partición. Las interpretaciones de errores son borradores sugeridos por IA: el estudiante debe comprobar cada mensaje, corregir las categorías y poder defender las conclusiones antes de entregar.
