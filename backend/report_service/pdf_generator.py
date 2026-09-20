"""Report Service - PDF 报告生成（支持 Markdown 表格和加粗）"""
import os
import re
import logging
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# 全局字体缓存
_font_registered = False
_font_name = 'Helvetica'


def _find_font():
    """查找可用的中文字体"""
    global _font_registered, _font_name
    if _font_registered:
        return _font_name

    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    font_candidates = [
        ("simhei.ttf", "SimHei"),
        ("msyh.ttc", "MSYH"),
        ("msyh.ttf", "MSYH2"),
        ("simkai.ttf", "SimKai"),
        ("simsun.ttc", "SimSun"),
        ("STHeiti Light.ttc", "STHeiti"),
        ("STHeiti Medium.ttc", "STHeitiM"),
    ]

    for filename, font_name in font_candidates:
        font_path = os.path.join(system_root, "Fonts", filename)
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                _font_registered = True
                _font_name = font_name
                logger.info(f"成功注册字体: {font_name}")
                return font_name
            except Exception as e:
                logger.warning(f"字体 {filename} 注册失败: {e}")
                continue

    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        _font_registered = True
        _font_name = 'STSong-Light'
        logger.info("使用内置 CID 字体: STSong-Light")
        return _font_name
    except Exception:
        pass

    logger.warning("未找到中文字体，将使用 Helvetica")
    _font_registered = True
    return _font_name


def _escape_xml(text: str) -> str:
    """转义 XML 特殊字符"""
    if not text:
        return ""
    text = text.replace("&amp;", "&")
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text


def _parse_markdown_bold(text: str) -> str:
    """将 Markdown **加粗** 转换为 reportlab 的 <b> 标签"""
    # 处理 **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 处理 *text* -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    return text


def _is_table_separator(line: str) -> bool:
    """检测是否为 Markdown 表格分隔行 |---|---|"""
    stripped = line.strip()
    if not stripped.startswith('|') or not stripped.endswith('|'):
        return False
    inner = stripped[1:-1]
    parts = [p.strip() for p in inner.split('|')]
    return all(re.match(r'^:?-+(:?)?$', p) for p in parts if p)


def _is_table_row(line: str) -> bool:
    """检测是否为 Markdown 表格行 | cell1 | cell2 |"""
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|') and '|' in stripped[1:-1]


def _parse_table_row(line: str) -> list:
    """解析表格行，返回单元格内容列表"""
    stripped = line.strip()
    inner = stripped[1:-1]  # 去掉首尾的 |
    cells = [cell.strip() for cell in inner.split('|')]
    return cells


def generate_pdf_report(
    report_text: str,
    topic: str,
    industry: str,
    horizon: str
) -> bytes:
    """生成PDF格式的投资报告，支持 Markdown 表格和加粗"""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    font_name = _find_font()
    styles = getSampleStyleSheet()

    # 定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8 * mm,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=6 * mm,
        alignment=TA_CENTER
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=15,
        spaceAfter=3 * mm,
        alignment=TA_JUSTIFY,
        wordWrap='CJK'
    )

    bold_body_style = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName=font_name,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=2 * mm,
        spaceAfter=2 * mm
    )

    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceBefore=8 * mm,
        spaceAfter=4 * mm
    )

    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=12,
        textColor=colors.HexColor('#3498db'),
        spaceBefore=6 * mm,
        spaceAfter=3 * mm
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        textColor=colors.white,
        alignment=TA_CENTER,
        leading=13
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8.5,
        leading=12,
        alignment=TA_LEFT,
        wordWrap='CJK'
    )

    story = []

    # 标题
    story.append(Paragraph("智能投研助手 - 投资分析报告", title_style))
    meta_text = f"研究主题: {_escape_xml(topic)} | 行业焦点: {_escape_xml(industry)} | 时间范围: {_escape_xml(horizon)}"
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(Spacer(2 * mm, 2 * mm))

    if not report_text:
        story.append(Paragraph("（无报告内容）", body_style))
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

    lines = report_text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            story.append(Spacer(1 * mm, 1 * mm))
            i += 1
            continue

        # 检测表格：连续的行都是表格格式
        if _is_table_row(line) and not _is_table_separator(line):
            table_rows = []
            # 收集表格所有行
            while i < len(lines) and _is_table_row(lines[i].strip()):
                row_line = lines[i].strip()
                if _is_table_separator(row_line):
                    i += 1
                    continue
                cells = _parse_table_row(row_line)
                # 处理每个单元格的加粗
                processed_cells = []
                for cell in cells:
                    cell = _parse_markdown_bold(_escape_xml(cell))
                    processed_cells.append(Paragraph(cell, table_cell_style))
                table_rows.append(processed_cells)
                i += 1

            if table_rows:
                # 计算列数
                col_count = max(len(row) for row in table_rows)
                # 补齐列数
                for row in table_rows:
                    while len(row) < col_count:
                        row.append(Paragraph("", table_cell_style))

                # 计算列宽
                available_width = A4[0] - 36 * mm
                col_width = available_width / col_count

                table = Table(table_rows, colWidths=[col_width] * col_count)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTNAME', (0, 0), (-1, 0), font_name),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('TOPPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                    ('TOPPADDING', (0, 1), (-1, -1), 6),
                ]))
                story.append(Spacer(2 * mm, 2 * mm))
                story.append(table)
                story.append(Spacer(2 * mm, 2 * mm))
            continue

        # 处理 Markdown 标题
        if line.startswith('# '):
            text = _parse_markdown_bold(_escape_xml(line[2:].strip()))
            story.append(Paragraph(text, heading1_style))
            i += 1
            continue
        elif line.startswith('## '):
            text = _parse_markdown_bold(_escape_xml(line[3:].strip()))
            story.append(Paragraph(text, heading2_style))
            i += 1
            continue
        elif line.startswith('### '):
            text = _parse_markdown_bold(_escape_xml(line[4:].strip()))
            story.append(Paragraph(text, heading2_style))
            i += 1
            continue

        # 处理数字标题，如 "1. 标题与摘要"
        if re.match(r'^\d+\.\s+', line):
            text = _parse_markdown_bold(_escape_xml(line))
            story.append(Paragraph(text, heading1_style))
            i += 1
            continue

        # 处理子标题，如 "4.1 机制维度"
        if re.match(r'^\d+\.\d+\s+', line):
            text = _parse_markdown_bold(_escape_xml(line))
            story.append(Paragraph(text, heading2_style))
            i += 1
            continue

        # 普通段落，处理加粗
        text = _parse_markdown_bold(_escape_xml(line))
        story.append(Paragraph(text, body_style))
        i += 1

    # 页脚
    story.append(Spacer(8 * mm, 8 * mm))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        textColor=colors.HexColor('#95a5a6'),
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 智能投研助手",
        footer_style
    ))

    # 构建 PDF
    try:
        doc.build(story)
    except Exception as e:
        logger.error(f"PDF build 失败: {e}")
        buffer.seek(0)
        buffer.truncate(0)
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        simple_style = ParagraphStyle('Simple', parent=styles['Normal'], fontName='Helvetica', fontSize=9)
        simple_story = [
            Paragraph("Investment Research Report", simple_style),
            Spacer(5 * mm, 5 * mm),
        ]
        for line in (report_text or "").split('\n')[:100]:
            simple_story.append(Paragraph(_escape_xml(line[:300]), simple_style))
        doc.build(simple_story)

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
