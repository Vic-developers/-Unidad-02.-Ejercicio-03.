"""Create the concise, evidence-based PDF summary from the generated reports."""
from __future__ import annotations

import json
import io
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.pdfgen import canvas as pdf_canvas
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Entrega_Ejercicio_03.pdf"

font_regular = Path("C:/Windows/Fonts/arial.ttf")
font_bold = Path("C:/Windows/Fonts/arialbd.ttf")
if font_regular.exists() and font_bold.exists():
    pdfmetrics.registerFont(TTFont("Arial", str(font_regular)))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_bold)))
    BODY, BOLD = "Arial", "Arial-Bold"
else:
    BODY, BOLD = "Helvetica", "Helvetica-Bold"

metrics = pd.read_csv(ROOT / "reports/text_metrics.csv")
class_metrics = pd.read_csv(ROOT / "reports/class_metrics.csv")
errors = pd.read_csv(ROOT / "reports/error_analysis.csv")
confusion_df = pd.read_csv(ROOT / "reports/confusion_matrix.csv", index_col=0)
audit = json.loads((ROOT / "reports/audit_summary.json").read_text(encoding="utf-8"))
confusion = confusion_df.to_numpy()
selected = str(metrics.iloc[0]["model"])
selected_row = metrics.iloc[0]
spam = class_metrics[class_metrics.label == "spam"].iloc[0]

# El documento usa una paleta neutra; las figuras se conservan a color.
navy, teal, pale, grey, orange = (colors.HexColor("#202020"), colors.HexColor("#303030"),
                                  colors.HexColor("#eeeeee"), colors.HexColor("#555555"),
                                  colors.HexColor("#888888"))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleES", fontName=BOLD, fontSize=24, leading=29,
    textColor=navy, alignment=TA_LEFT, spaceAfter=7))
styles.add(ParagraphStyle(name="SubES", fontName=BODY, fontSize=9, leading=13,
    textColor=grey, spaceAfter=8))
styles.add(ParagraphStyle(name="H2ES", fontName=BOLD, fontSize=12, leading=15,
    textColor=teal, spaceBefore=7, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyES", fontName=BODY, fontSize=9.1, leading=12.6,
    textColor=navy, alignment=TA_JUSTIFY, spaceAfter=4))
styles.add(ParagraphStyle(name="SmallES", fontName=BODY, fontSize=7.8, leading=10.2,
    textColor=grey, alignment=TA_JUSTIFY, spaceAfter=2))
styles.add(ParagraphStyle(name="CellES", fontName=BODY, fontSize=8.1, leading=10,
    textColor=navy, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="CellHeadES", fontName=BOLD, fontSize=8.1, leading=10,
    textColor=colors.white, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name="KPI", fontName=BOLD, fontSize=15, leading=17,
    textColor=navy, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="KPIlabel", fontName=BODY, fontSize=7.3, leading=9,
    textColor=grey, alignment=TA_CENTER))

def P(text: str, style: str = "BodyES") -> Paragraph:
    return Paragraph(text, styles[style])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#bbbbbb"))
    canvas.line(0.55 * inch, 0.43 * inch, 7.95 * inch, 0.43 * inch)
    canvas.setFont(BODY, 7.5)
    canvas.setFillColor(grey)
    canvas.drawString(0.57 * inch, 0.25 * inch, "INF8239 · Ejercicio 03 · UCI SMS Spam Collection")
    canvas.drawRightString(7.93 * inch, 0.25 * inch, f"{doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.58*inch,
    leftMargin=0.58*inch, topMargin=0.5*inch, bottomMargin=0.58*inch,
    title="Ejercicio 03 - Corpus público y clasificador reproducible",
    author="Victor Elvis Lorenzo")
story = []

story.append(P("Ejercicio 03", "TitleES"))
story.append(P("Corpus público y clasificador de texto reproducible", "SubES"))
story.append(P("Autor: Victor Elvis Lorenzo &nbsp;&nbsp; | &nbsp;&nbsp; Ejecución: 20 octubre 2026", "SmallES"))
story.append(P('Repositorio de entrega: <link href="https://github.com/Vic-developers/-Unidad-02.-Ejercicio-03.." color="#000000">github.com/Vic-developers/-Unidad-02.-Ejercicio-03..</link>.', "SmallES"))
story.append(Spacer(1, 5))
story.append(P("Pregunta", "H2ES"))
story.append(P("¿Cómo se comparan TF-IDF con Complement Naive Bayes y regresión logística para clasificar SMS legítimos (ham) y spam cuando se retiran textos duplicados antes de separar entrenamiento y prueba?", "BodyES"))

kpis = [[P("5,572", "KPI"), P("415", "KPI"), P("3,867 / 1,290", "KPI"), P(f"{selected_row['f1_macro']:.3f}", "KPI")],
        [P("mensajes auditados", "KPIlabel"), P("repeticiones retiradas", "KPIlabel"), P("entrenamiento / prueba", "KPIlabel"), P("F1 macro del mejor", "KPIlabel")]]
table = Table(kpis, colWidths=[1.8*inch]*4, rowHeights=[0.35*inch,0.3*inch])
table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),pale), ("BOX",(0,0),(-1,-1),0.5,colors.HexColor("#bbbbbb")), ("INNERGRID",(0,0),(-1,-1),0.35,colors.HexColor("#bbbbbb")), ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),4), ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
story.extend([table, Spacer(1, 6)])

story.append(P("Corpus y decisiones", "H2ES"))
story.append(P('SMS Spam Collection, UCI Machine Learning Repository, DOI <link href="https://doi.org/10.24432/C5CC84" color="#000000">10.24432/C5CC84</link>, licencia CC BY 4.0. El ZIP y su huella SHA-256 se incluyen en la entrega. La colección contiene mensajes en inglés con etiquetas ham/spam.', "BodyES"))
story.append(P(f"Distribución: 4,825 ham y 747 spam. Sin faltantes ni etiquetas en conflicto. Se detectaron {audit['duplicate_rows_exact']} duplicados exactos y 12 coincidencias adicionales tras normalizar mayúsculas/espacios. Se conservó una fila por texto normalizado antes del split; el split fue estratificado 75/25 con semilla 42.", "BodyES"))
story.append(P("Comparación en el mismo split", "H2ES"))
rows = [[P("Pipeline", "CellHeadES"), P("F1 macro", "CellHeadES"), P("Exactitud", "CellHeadES")]]
name_display = {"logistic_regression": "TF-IDF + regresión logística", "complement_nb": "TF-IDF + Complement Naive Bayes", "dummy": "Baseline Dummy mayoritario"}
for _, row in metrics.iterrows():
    rows.append([P(name_display.get(row.model, str(row.model)), "CellES"),
                 P(f"{row.f1_macro:.6f}", "CellES"), P(f"{row.accuracy:.6f}", "CellES")])
tbl = Table(rows, colWidths=[4.2*inch,1.2*inch,1.2*inch], repeatRows=1)
tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),navy), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,pale]), ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#bbbbbb")), ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("ALIGN",(1,1),(-1,-1),"RIGHT"), ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
story.append(tbl)
story.append(Spacer(1, 4))
story.append(P(f"Resultado: {name_display.get(selected, selected)} fue el mejor en esta partición. En spam, precision={spam['precision']:.3f}, recall={spam['recall']:.3f} y F1={spam['f1']:.3f}; quedan 9 falsos negativos y 14 falsos positivos.", "BodyES"))
story.append(Image(str(ROOT / "reports/corpus_profile.png"), width=6.75*inch, height=2.28*inch))

story.append(PageBreak())
story.append(P("Evaluación y lectura de errores", "TitleES"))
story.append(P("Resultados observados en una nueva ejecución local; las cifras del informe previo no se reutilizaron como resultados ejecutados.", "SubES"))
story.append(P("Evidencia de la prueba", "H2ES"))
story.append(P(f"El holdout incluye 1,290 mensajes: 1,129 ham y 161 spam. Matriz (filas = clase real ham/spam; columnas = predicción ham/spam): [[1115, 14], [9, 152]]. El F1 macro de regresión logística es {selected_row['f1_macro']:.6f}, frente a {metrics.loc[metrics.model=='complement_nb','f1_macro'].iloc[0]:.6f} para Complement Naive Bayes y 0.466722 para Dummy.", "BodyES"))
story.append(Image(str(ROOT / "reports/confusion_matrix.png"), width=4.75*inch, height=3.02*inch, hAlign="CENTER"))
story.append(P("Revisión cualitativa", "H2ES"))
false_positive = int(confusion[0, 1])
false_negative = int(confusion[1, 0])
sample_fp = int(((errors.real == "ham") & (errors.predicted == "spam")).sum())
sample_fn = int(((errors.real == "spam") & (errors.predicted == "ham")).sum())
sample_size = int(len(errors))
total_errors = false_positive + false_negative
story.append(P(f"El modelo produjo {total_errors} errores en la prueba; se seleccionaron {sample_size} para análisis: {sample_fp} ham predichos como spam y {sample_fn} spam predichos como ham. Las notas observan mensajes personales muy breves, ambigüedad de expresiones como free/message/offer, promociones con estilo conversacional, y contexto o humor ausente. La tabla está en reports/error_analysis.csv. Las 20 notas son preliminares, requieren revisión del estudiante y no estiman frecuencias poblacionales.", "BodyES"))

story.append(P("Interpretación y decisión", "H2ES"))
story.append(P("La regresión logística superó a Complement Naive Bayes por una diferencia pequeña de 0.0064 puntos de F1 macro. El recall de spam fue 0.944 (9 omisiones). Se conserva como mejor modelo de esta prueba, pero el resultado es un prototipo: un único split del corpus histórico en inglés no permite afirmar rendimiento en mensajes actuales, español dominicano u otros canales.", "BodyES"))

story.append(P("Reproducibilidad y uso de IA", "H2ES"))
story.append(P("El README documenta instalación, dependencias, pruebas y ejecución. La notebook guarda salidas; el código vuelve a calcular los reportes y las figuras. Se ejecutaron 5 pruebas automatizadas, todas aprobadas. Codex se utilizó como apoyo para corregir errores de código, proponer pruebas y revisar la redacción; también ayudó a preparar notas preliminares para interpretar errores. Prompt relevante: integrar LAB04 y LAB05 en una entrega ejecutable; la consigna exige que pueda correrse desde las instrucciones del repositorio. Se verificaron descarga, SHA-256, conteos, partición, métricas, matriz y tests. El estudiante debe revisar las 20 notas y poder explicar el trabajo. El repositorio enlazado contiene los archivos de esta entrega.", "BodyES"))
story.append(P("Fuentes: UCI Machine Learning Repository, SMS Spam Collection (CC BY 4.0), https://archive.ics.uci.edu/dataset/228/sms+spam+collection. Almeida, Hidalgo y Yamakami (2011), DOI 10.1145/2034691.2034742. Consulte README.md y docs/AI_USAGE.md para instrucciones y declaración ampliada.", "SmallES"))

doc.build(story, onFirstPage=footer, onLaterPages=footer)

# Portada institucional centrada, con el escudo suministrado para esta entrega.
cover_buffer = io.BytesIO()
cover = pdf_canvas.Canvas(cover_buffer, pagesize=letter)
page_width, page_height = letter
seal = ROOT / "assets/uasd_seal.png"
seal_width, seal_height = 1.12 * inch, 1.40 * inch
cover.drawImage(str(seal), (page_width-seal_width)/2, 9.12*inch,
                width=seal_width, height=seal_height, preserveAspectRatio=True,
                anchor="c", mask="auto")
cover.setFillColor(colors.HexColor("#202020"))
cover.setFont(BOLD, 14)
cover.drawCentredString(page_width/2, 8.58*inch,
                        "UNIVERSIDAD AUTÓNOMA DE SANTO DOMINGO (UASD)")

def cover_section(label: str, value: str, y: float, value_size: float = 13):
    cover.setFont(BOLD, 10)
    cover.drawCentredString(page_width/2, y, label)
    cover.setFont(BODY, value_size)
    cover.drawCentredString(page_width/2, y-20, value)

cover_section("ASIGNATURA", "Ciencia de Datos II", 7.42*inch, 14)
cover_section("TEMA", "Unidad 02 - Ejercicio 03", 6.62*inch, 14)
cover.setFont(BODY, 12)
cover.drawCentredString(page_width/2, 6.05*inch,
                        "Corpus público y clasificador de texto reproducible")
cover_section("PARTICIPANTE", "Victor Elvis Lorenzo", 4.86*inch, 13)
cover_section("PROFESOR", "Edwin Ramón José Nolasco", 3.80*inch, 13)
cover.showPage()
cover.save()
cover_buffer.seek(0)

body_reader = PdfReader(str(OUT))
cover_reader = PdfReader(cover_buffer)
writer = PdfWriter()
writer.add_page(cover_reader.pages[0])
for page in body_reader.pages:
    writer.add_page(page)
writer.add_metadata({"/Title": "Ejercicio 03 - Corpus público y clasificador de texto reproducible",
                     "/Author": "Victor Elvis Lorenzo"})
temp_out = OUT.with_name(OUT.stem + ".tmp.pdf")
with temp_out.open("wb") as target:
    writer.write(target)
temp_out.replace(OUT)
print(f"PDF creado con portada: {OUT}")
