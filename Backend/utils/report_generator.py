from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import os

def generate_report(username, prediction, confidence, input_type):

    # Create reports folder if not exists
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    file_name = f"{username}_report.pdf"
    file_path = os.path.join(reports_dir, file_name)

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []

    # Add Logo
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=100, height=50)
        content.append(logo)

    content.append(Spacer(1, 10))
    content.append(Paragraph("<b>Parkinson's Disease Detection Report</b>", styles['Title']))
    content.append(Spacer(1, 20))

    # Patient Info Table
    data = [
        ["Patient Name", username],
        ["Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ["Test Type", input_type]
    ]
    table = Table(data, colWidths=[150, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    content.append(table)
    content.append(Spacer(1, 20))

    # Prediction Table
    result_data = [
        ["Prediction", prediction],
        ["Confidence", f"{confidence}%"]
    ]
    result_table = Table(result_data, colWidths=[150, 250])
    result_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))
    content.append(result_table)
    content.append(Spacer(1, 20))

    # Remark
    remark = "⚠️ Signs detected. Please consult a neurologist." if prediction == "Parkinson's" else "✅ No significant signs detected."
    content.append(Paragraph("<b>Remark:</b>", styles['Heading2']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(remark, styles['Normal']))
    content.append(Spacer(1, 30))

    # Footer
    content.append(Paragraph("This is an AI-generated report for early screening purposes only.", styles['Italic']))

    doc.build(content)

    return file_path