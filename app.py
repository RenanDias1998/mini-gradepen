import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image


CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10
TOTAL_ALTERNATIVES = 5
MARKER_CANVAS = (900, 1300)
DEFAULT_GRID = {
    "columns": np.array([218, 354, 490, 626, 762], dtype=np.int32),
    "rows": np.array([282, 376, 469, 563, 657, 751, 844, 938, 1032, 1126], dtype=np.int32),
}
EXAMS_FILE = Path("provas.json")


def init_state():
    defaults = {
        "draft_answer_key": [""] * TOTAL_QUESTIONS,
        "saved_answer_key": [],
        "saved_answer_key_version": 0,
        "class_results": [],
        "last_processed": None,
        "active_exam_code": "",
        "active_exam": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def pil_to_bgr(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def answer_key_complete(answer_key):
    return len(answer_key) == TOTAL_QUESTIONS and all(answer_key)


def slugify_exam_code(text):
    allowed = []
    for char in text.strip().upper():
        if char.isalnum():
            allowed.append(char)
        elif char in ("-", "_", "/"):
            allowed.append("-")
        elif char.isspace():
            allowed.append("-")
    slug = "".join(allowed).replace("--", "-").strip("-")
    return slug or "PROVA"


def load_exams():
    if not EXAMS_FILE.exists():
        return {}
    try:
        return json.loads(EXAMS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_exams(exams):
    EXAMS_FILE.write_text(json.dumps(exams, ensure_ascii=False, indent=2), encoding="utf-8")


def build_exam_url(base_url, exam_code):
    clean_base = base_url.strip().rstrip("/")
    if not clean_base:
        return ""
    return f"{clean_base}?{urlencode({'prova': exam_code})}"


def build_qr_image(data):
    qr = qrcode.QRCode(version=2, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def qr_image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def set_active_exam(exam):
    st.session_state.active_exam = exam
    st.session_state.active_exam_code = exam["code"]
    st.session_state.saved_answer_key = exam["answer_key"]
    st.session_state.saved_answer_key_version = exam.get("version", 1)
    st.session_state.draft_answer_key = exam["answer_key"].copy()


def sync_exam_from_query(exams):
    query_params = st.query_params
    exam_code = query_params.get("prova", "")
    if not exam_code:
        return

    exam_code = slugify_exam_code(exam_code)
    exam = exams.get(exam_code)
    if exam is None:
        st.warning(f"O link da prova '{exam_code}' nao foi encontrado no cadastro local.")
        return

    if st.session_state.active_exam_code != exam_code:
        set_active_exam(exam)


def order_points(points):
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def four_point_transform(image, points, size=None):
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    if size is None:
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_width = max(int(width_a), int(width_b))
        max_height = max(int(height_a), int(height_b))
    else:
        max_width, max_height = size

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped, rect


def resize_for_preview(image, max_width=1100):
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy(), 1.0
    scale = max_width / width
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def preprocess_gray(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return enhanced, blurred


def detect_sheet(image):
    preview, scale = resize_for_preview(image)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    _, blurred = preprocess_gray(gray)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    chosen = None
    for contour in contours[:15]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > preview.shape[0] * preview.shape[1] * 0.15:
            chosen = approx.reshape(4, 2).astype(np.float32) / scale
            break

    if chosen is None and contours:
        rect = cv2.minAreaRect(contours[0])
        chosen = cv2.boxPoints(rect).astype(np.float32) / scale

    if chosen is None:
        raise ValueError("Nao foi possivel detectar a folha automaticamente.")

    warped, ordered = four_point_transform(image, chosen)
    preview_outline = preview.copy()
    preview_points = (order_points(chosen) * scale).astype(np.int32)
    cv2.polylines(preview_outline, [preview_points], True, (0, 255, 0), 4)

    return warped, {"folha_detectada": preview_outline, "bordas_folha": edges, "pontos_folha": ordered}


def threshold_for_dark_regions(gray):
    enhanced, blurred = preprocess_gray(gray)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        11,
    )
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    merged = cv2.bitwise_or(adaptive, otsu)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=1)
    return enhanced, merged


def sort_markers(markers):
    ordered = sorted(markers, key=lambda marker: marker["center"][1])
    top = sorted(ordered[:2], key=lambda marker: marker["center"][0])
    bottom = sorted(ordered[2:], key=lambda marker: marker["center"][0])
    return [top[0], top[1], bottom[1], bottom[0]]


def find_reference_markers(sheet):
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
    enhanced, dark = threshold_for_dark_regions(gray)
    height, width = dark.shape
    search = np.zeros_like(dark)
    search[:, int(width * 0.48):] = dark[:, int(width * 0.48):]

    contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > height * width * 0.04:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(h)
        if not 0.7 <= aspect <= 1.3:
            continue

        fill_ratio = area / float(w * h)
        if fill_ratio < 0.65:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) < 4:
            continue

        candidates.append(
            {
                "rect": (x, y, w, h),
                "center": (x + w / 2.0, y + h / 2.0),
                "area": area,
            }
        )

    if len(candidates) < 4:
        raise ValueError("Os quadrados pretos de referencia nao foram encontrados.")

    candidates.sort(key=lambda item: item["area"], reverse=True)
    best_group = None

    for start in range(len(candidates)):
        for end in range(start + 4, len(candidates) + 1):
            group = candidates[start:end]
            if len(group) < 4:
                continue

            centers = np.array([item["center"] for item in group], dtype=np.float32)
            median_x = np.median(centers[:, 0])
            top = centers[centers[:, 1] <= np.median(centers[:, 1])]
            bottom = centers[centers[:, 1] > np.median(centers[:, 1])]
            left = centers[centers[:, 0] <= median_x]
            right = centers[centers[:, 0] > median_x]

            if len(top) >= 2 and len(bottom) >= 2 and len(left) >= 2 and len(right) >= 2:
                best_group = sorted(group, key=lambda item: item["area"], reverse=True)[:4]
                break
        if best_group:
            break

    if best_group is None:
        best_group = candidates[:4]

    ordered_markers = sort_markers(best_group)
    tl, tr, br, bl = ordered_markers
    marker_points = np.array(
        [
            [tl["rect"][0], tl["rect"][1]],
            [tr["rect"][0] + tr["rect"][2], tr["rect"][1]],
            [br["rect"][0] + br["rect"][2], br["rect"][1] + br["rect"][3]],
            [bl["rect"][0], bl["rect"][1] + bl["rect"][3]],
        ],
        dtype=np.float32,
    )
    marker_debug = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    for index, marker in enumerate(best_group, start=1):
        x, y, w, h = marker["rect"]
        cv2.rectangle(marker_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = map(int, marker["center"])
        cv2.putText(
            marker_debug,
            str(index),
            (cx - 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return marker_points, {"marcadores": marker_debug, "mascara_escura": dark}


def infer_positions(binary):
    height, width = binary.shape

    col_window = binary[int(height * 0.18): int(height * 0.88), int(width * 0.15): int(width * 0.92)]
    vertical_projection = col_window.sum(axis=0).astype(np.float32)
    col_indices = np.where(vertical_projection > vertical_projection.max() * 0.45)[0]

    if len(col_indices) > 0:
        groups = np.split(col_indices, np.where(np.diff(col_indices) > 12)[0] + 1)
        centers = np.array(
            [int(group.mean()) + int(width * 0.15) for group in groups if len(group) > 4],
            dtype=np.int32,
        )
    else:
        centers = np.array([], dtype=np.int32)

    if len(centers) != TOTAL_ALTERNATIVES:
        centers = DEFAULT_GRID["columns"]

    row_window = binary[int(height * 0.18): int(height * 0.92), int(width * 0.18): int(width * 0.86)]
    horizontal_projection = row_window.sum(axis=1).astype(np.float32)
    row_indices = np.where(horizontal_projection > horizontal_projection.max() * 0.55)[0]

    if len(row_indices) > 0:
        groups = np.split(row_indices, np.where(np.diff(row_indices) > 14)[0] + 1)
        rows = np.array(
            [int(group.mean()) + int(height * 0.18) for group in groups if len(group) > 4],
            dtype=np.int32,
        )
    else:
        rows = np.array([], dtype=np.int32)

    if len(rows) != TOTAL_QUESTIONS:
        rows = DEFAULT_GRID["rows"]

    return centers, rows


def score_bubbles(warped_answers):
    gray = cv2.cvtColor(warped_answers, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    adaptive = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    columns, rows = infer_positions(cleaned)
    radius = 28
    scores = []
    answers = []

    for center_y in rows:
        row_scores = []
        for center_x in columns:
            mask = np.zeros_like(cleaned)
            cv2.circle(mask, (int(center_x), int(center_y)), radius, 255, -1)
            focused = cv2.bitwise_and(cleaned, cleaned, mask=mask)
            dark_pixels = cv2.countNonZero(focused)
            area = cv2.countNonZero(mask)
            row_scores.append(dark_pixels / float(max(area, 1)))

        scores.append(row_scores)
        best_index = int(np.argmax(row_scores))
        sorted_scores = np.sort(np.array(row_scores))[::-1]
        confident = sorted_scores[0] > 0.18 and (sorted_scores[0] - sorted_scores[1] > 0.035)
        answers.append(CHOICES[best_index] if confident else "?")

    diagnostics = {
        "binaria_bolhas": cleaned,
        "colunas": columns,
        "linhas": rows,
        "scores": scores,
    }
    return answers, diagnostics


def build_answer_overlay(warped_answers, answers, diagnostics):
    overlay = warped_answers.copy()
    columns = diagnostics["colunas"]
    rows = diagnostics["linhas"]
    scores = diagnostics["scores"]

    for question_index, center_y in enumerate(rows):
        cv2.putText(
            overlay,
            f"Q{question_index + 1}",
            (28, int(center_y + 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        for choice_index, center_x in enumerate(columns):
            selected = answers[question_index] == CHOICES[choice_index]
            color = (0, 180, 0) if selected else (0, 140, 255)
            cv2.circle(overlay, (int(center_x), int(center_y)), 30, color, 2)
            cv2.putText(
                overlay,
                f"{scores[question_index][choice_index]:.2f}",
                (int(center_x - 28), int(center_y - 38)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    return overlay


def extract_answer_area(sheet):
    marker_points, marker_diagnostics = find_reference_markers(sheet)
    warped_answers, _ = four_point_transform(sheet, marker_points, size=MARKER_CANVAS)
    preview = sheet.copy()
    cv2.polylines(preview, [marker_points.astype(np.int32)], True, (255, 0, 0), 4)

    return warped_answers, {
        "area_gabarito": preview,
        **marker_diagnostics,
    }


def process_answer_sheet(image_bgr):
    sheet, sheet_diagnostics = detect_sheet(image_bgr)
    answer_area, marker_diagnostics = extract_answer_area(sheet)
    answers, bubble_diagnostics = score_bubbles(answer_area)
    overlay = build_answer_overlay(answer_area, answers, bubble_diagnostics)

    return {
        "folha": sheet,
        "gabarito": answer_area,
        "respostas": answers,
        "diagnosticos": {
            **sheet_diagnostics,
            **marker_diagnostics,
            **bubble_diagnostics,
            "overlay_leitura": overlay,
        },
    }


def render_diagnostics(result):
    st.subheader("Diagnostico visual")
    col1, col2 = st.columns(2)
    with col1:
        st.image(bgr_to_rgb(result["diagnosticos"]["folha_detectada"]), caption="Folha detectada")
        st.image(result["diagnosticos"]["bordas_folha"], caption="Bordas usadas na deteccao")
        st.image(bgr_to_rgb(result["diagnosticos"]["marcadores"]), caption="Quadrados pretos encontrados")
    with col2:
        st.image(bgr_to_rgb(result["diagnosticos"]["area_gabarito"]), caption="Recorte bruto do bloco de respostas")
        st.image(bgr_to_rgb(result["gabarito"]), caption="Gabarito alinhado")
        st.image(bgr_to_rgb(result["diagnosticos"]["overlay_leitura"]), caption="Leitura com scores por bolha")

    st.image(result["diagnosticos"]["binaria_bolhas"], caption="Mascara binaria usada na leitura")


def select_answer(question_index, choice):
    st.session_state.draft_answer_key[question_index] = choice


def render_answer_key_editor():
    st.subheader("Gabarito do professor")
    st.caption("Clique na alternativa de cada questao. A escolha permanece ate voce salvar.")

    for question_index in range(TOTAL_QUESTIONS):
        current_choice = st.session_state.draft_answer_key[question_index]
        row_columns = st.columns([1.2, 1, 1, 1, 1, 1, 1.6])
        row_columns[0].markdown(f"**Q{question_index + 1}**")

        for choice_index, choice in enumerate(CHOICES, start=1):
            selected = current_choice == choice
            button_type = "primary" if selected else "secondary"
            row_columns[choice_index].button(
                choice,
                key=f"draft_q{question_index}_{choice}",
                type=button_type,
                use_container_width=True,
                on_click=select_answer,
                args=(question_index, choice),
            )

        if current_choice:
            row_columns[6].markdown(
                f"Selecionada: <span style='color:#15803d; font-weight:700;'>{current_choice}</span>",
                unsafe_allow_html=True,
            )
        else:
            row_columns[6].markdown(
                "<span style='color:#b45309; font-weight:600;'>Aguardando escolha</span>",
                unsafe_allow_html=True,
            )


def compute_score(student_answers, saved_answer_key):
    comparisons = []
    correct = 0

    for question_index, official_answer in enumerate(saved_answer_key):
        student_answer = student_answers[question_index] if question_index < len(student_answers) else "?"
        is_correct = student_answer == official_answer
        correct += int(is_correct)
        comparisons.append(
            {
                "questao": question_index + 1,
                "gabarito": official_answer,
                "aluno": student_answer,
                "status": "Acerto" if is_correct else "Erro",
            }
        )

    return correct, comparisons


def build_student_record(name, class_name, exam, student_answers, correct_answers, comparisons):
    record = {
        "Prova": exam["title"],
        "Codigo Prova": exam["code"],
        "Data Prova": exam.get("date", ""),
        "Aluno": name or "Nao informado",
        "Turma": class_name or "Nao informada",
        "Acertos": correct_answers,
        "Erros": TOTAL_QUESTIONS - correct_answers,
        "Respostas Lidas": " ".join(student_answers),
        "Gabarito Usado": " ".join(exam["answer_key"]),
        "Versao Gabarito": exam.get("version", 1),
    }

    for item in comparisons:
        record[f"Q{item['questao']}"] = item["aluno"]
        record[f"Q{item['questao']}_status"] = item["status"]

    return record


def export_results_to_excel():
    if not st.session_state.class_results:
        return None

    data = pd.DataFrame(st.session_state.class_results)
    ordered_columns = [
        "Prova",
        "Codigo Prova",
        "Data Prova",
        "Aluno",
        "Turma",
        "Acertos",
        "Erros",
        "Respostas Lidas",
        "Gabarito Usado",
        "Versao Gabarito",
    ]

    for question_index in range(1, TOTAL_QUESTIONS + 1):
        ordered_columns.append(f"Q{question_index}")
        ordered_columns.append(f"Q{question_index}_status")

    data = data[[column for column in ordered_columns if column in data.columns]]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="Correcoes")

    output.seek(0)
    return output


def render_saved_results():
    st.subheader("Turma corrigida")
    if not st.session_state.class_results:
        st.info("Nenhum aluno foi salvo ainda.")
        return

    dataframe = pd.DataFrame(st.session_state.class_results)
    st.dataframe(dataframe, use_container_width=True)

    excel_file = export_results_to_excel()
    if excel_file is not None:
        st.download_button(
            "Baixar Excel da turma",
            data=excel_file,
            file_name="correcoes_turma.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def generate_printable_template(exam, exam_url):
    qr_b64 = ""
    if exam_url:
        qr_b64 = qr_image_to_base64(build_qr_image(exam_url))

    qr_html = (
        f'<img src="data:image/png;base64,{qr_b64}" alt="QR da prova" style="width:36mm;height:36mm;display:block;" />'
        if qr_b64
        else '<div style="font-size:9px;color:#666;text-align:center;">Defina a URL publica do app para gerar o QR.</div>'
    )

    rows_html = []
    for question_index in range(TOTAL_QUESTIONS):
        bubbles = "".join('<div class="bubble"></div>' for _ in CHOICES)
        rows_html.append(
            f'<div class="answer-row"><div class="question-label">Q{question_index + 1}</div>{bubbles}</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>{exam["title"]}</title>
  <style>
    :root {{
      --page-width: 210mm;
      --page-height: 297mm;
      --ink: #111111;
      --soft-ink: #444444;
      --line: #bfc5cc;
      --paper: #ffffff;
      --panel: #f6f7f8;
      --marker-size: 8mm;
      --bubble-size: 5.6mm;
      --row-gap: 3.5mm;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eceff1; font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); }}
    .page {{ width: var(--page-width); min-height: var(--page-height); margin: 12px auto; background: var(--paper); padding: 8mm 8mm 9mm; }}
    .header {{ border: 1px solid var(--line); padding: 3.5mm 4.5mm; margin-bottom: 4mm; }}
    .header-top {{ display:flex; justify-content:space-between; gap:12px; margin-bottom:2.5mm; }}
    .title {{ font-size:18px; font-weight:700; }}
    .subtitle {{ font-size:10px; color:var(--soft-ink); }}
    .meta-grid {{ display:grid; grid-template-columns:1.2fr 0.9fr 0.9fr; gap:3mm; }}
    .field {{ border:1px solid var(--line); padding:2.2mm; min-height:12mm; }}
    .field-label {{ font-size:10px; font-weight:700; text-transform:uppercase; color:var(--soft-ink); margin-bottom:1.2mm; }}
    .field-line {{ border-bottom:1px solid #949aa1; height:5mm; }}
    .content {{ display:grid; grid-template-columns:58mm 1fr; gap:5mm; align-items:start; }}
    .left-panel, .right-panel {{ border:1px solid var(--line); min-height:170mm; position:relative; }}
    .left-panel {{ padding:4mm; background:linear-gradient(var(--panel), var(--panel)) top/100% 14mm no-repeat, var(--paper); }}
    .panel-title {{ font-size:12px; font-weight:700; text-transform:uppercase; color:var(--soft-ink); margin-bottom:3mm; }}
    .qr-box {{ width:42mm; height:42mm; border:1px solid var(--ink); margin:0 auto 3mm; display:grid; place-items:center; background:#fff; }}
    .id-box {{ border:1px solid var(--line); padding:2.5mm; margin-top:3mm; min-height:16mm; }}
    .instructions {{ margin-top:4mm; padding-left:5mm; font-size:9.5px; line-height:1.3; }}
    .instructions li + li {{ margin-top:1.2mm; }}
    .right-panel {{ padding:6mm 6mm 7mm; }}
    .marker {{ position:absolute; width:var(--marker-size); height:var(--marker-size); background:#000; }}
    .marker.tl {{ top:5mm; left:5mm; }}
    .marker.tr {{ top:5mm; right:5mm; }}
    .marker.bl {{ bottom:5mm; left:5mm; }}
    .marker.br {{ bottom:5mm; right:5mm; }}
    .answer-frame {{ border:1px solid #d1d6db; min-height:155mm; padding:7mm 6mm 6mm; }}
    .answer-guide {{ text-align:center; font-size:10px; color:var(--soft-ink); margin-bottom:4mm; font-weight:600; }}
    .answer-header, .answer-row {{ display:grid; grid-template-columns:15mm repeat(5, 1fr); align-items:center; column-gap:2mm; }}
    .answer-header {{ font-size:11px; font-weight:700; text-align:center; margin-bottom:2.5mm; padding:0 2mm; }}
    .answer-row {{ min-height:8mm; margin-bottom:var(--row-gap); padding:0 2mm; }}
    .question-label {{ font-size:11px; font-weight:700; }}
    .bubble {{ width:var(--bubble-size); height:var(--bubble-size); border:1.2px solid #000; border-radius:50%; margin:0 auto; background:#fff; }}
    .footer {{ margin-top:4mm; border:1px solid var(--line); padding:3mm 4mm; font-size:9.5px; color:var(--soft-ink); display:flex; justify-content:space-between; gap:8mm; flex-wrap:wrap; }}
    @page {{ size:A4 portrait; margin:0; }}
  </style>
</head>
<body>
  <main class="page">
    <section class="header">
      <div class="header-top">
        <div class="title">{exam["title"]}</div>
        <div class="subtitle">Codigo: {exam["code"]} | Data: {exam.get("date", "") or "-"}</div>
      </div>
      <div class="meta-grid">
        <div class="field"><div class="field-label">Nome do aluno</div><div class="field-line"></div></div>
        <div class="field"><div class="field-label">Turma</div><div class="field-line"></div></div>
        <div class="field"><div class="field-label">Codigo da prova</div><div class="field-line"></div></div>
      </div>
    </section>
    <section class="content">
      <aside class="left-panel">
        <div class="panel-title">Identificacao</div>
        <div class="qr-box">{qr_html}</div>
        <div class="id-box">
          <div class="field-label">Link da prova</div>
          <div style="font-size:8px; word-break:break-word;">{exam_url or "Defina a URL publica do app para gerar o link."}</div>
        </div>
        <ol class="instructions">
          <li>Preencha somente uma alternativa por questao.</li>
          <li>Pinte a bolha completamente com caneta preta ou azul escura.</li>
          <li>Nao dobre a folha e nao escreva dentro da area do gabarito.</li>
          <li>Mantenha os quadrados pretos livres para alinhamento automatico.</li>
        </ol>
      </aside>
      <section class="right-panel">
        <div class="marker tl"></div>
        <div class="marker tr"></div>
        <div class="marker bl"></div>
        <div class="marker br"></div>
        <div class="answer-frame">
          <div class="answer-guide">Marque o gabarito preenchendo completamente a bolha escolhida</div>
          <div class="answer-header"><div></div><div>A</div><div>B</div><div>C</div><div>D</div><div>E</div></div>
          {"".join(rows_html)}
        </div>
      </section>
    </section>
    <footer class="footer">
      <div>Gabarito vinculado a prova cadastrada no site.</div>
      <div>QR e link apontam para a consulta automatica desta prova.</div>
    </footer>
  </main>
</body>
</html>"""


def render_exam_registry():
    st.subheader("Cadastro da prova")
    st.caption("Salve a prova no site para gerar um link fixo e um QR code reutilizavel.")

    exam_title = st.text_input("Titulo da prova", key="exam_title_input")
    exam_date = st.text_input("Data da prova", key="exam_date_input", value=datetime.now().strftime("%Y-%m-%d"))
    exam_code_raw = st.text_input("Codigo da prova", key="exam_code_input", help="Exemplo: MAT-2026-04-22")
    base_url = st.text_input(
        "URL publica do app",
        key="base_url_input",
        help="Exemplo: https://seu-app.streamlit.app",
    )

    render_answer_key_editor()

    exam_code = slugify_exam_code(exam_code_raw or exam_title or "PROVA")
    if exam_code_raw and exam_code_raw != exam_code:
        st.caption(f"Codigo normalizado para uso na URL: `{exam_code}`")

    action_col1, action_col2 = st.columns([1.2, 1.8])
    exams = load_exams()

    if action_col1.button("Salvar prova cadastrada", type="primary", use_container_width=True):
        if not exam_title.strip():
            st.warning("Informe um titulo para a prova.")
        elif not answer_key_complete(st.session_state.draft_answer_key):
            st.warning("Preencha todas as questoes antes de salvar.")
        else:
            current_version = exams.get(exam_code, {}).get("version", 0) + 1
            exam = {
                "code": exam_code,
                "title": exam_title.strip(),
                "date": exam_date.strip(),
                "answer_key": st.session_state.draft_answer_key.copy(),
                "version": current_version,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            exams[exam_code] = exam
            save_exams(exams)
            set_active_exam(exam)
            st.success("Prova salva. Agora o link e o QR code podem ser usados para abrir esta prova.")

    if action_col2.button("Carregar prova cadastrada selecionada", use_container_width=True):
        selected_code = st.session_state.get("selected_exam_code", "")
        if selected_code and selected_code in exams:
            set_active_exam(exams[selected_code])
            st.success(f"Prova '{selected_code}' carregada.")

    if exams:
        codes = list(exams.keys())
        selected_code = st.selectbox(
            "Provas cadastradas",
            options=[""] + codes,
            format_func=lambda code: "Selecione uma prova" if not code else f"{code} - {exams[code]['title']}",
            key="selected_exam_code",
        )
        if selected_code:
            selected_exam = exams[selected_code]
            st.caption(
                f"Ultima atualizacao: {selected_exam.get('updated_at', '-')} | Gabarito: {' '.join(selected_exam['answer_key'])}"
            )
    else:
        st.info("Nenhuma prova cadastrada ainda.")

    active_exam = st.session_state.active_exam
    if active_exam:
        exam_url = build_exam_url(base_url, active_exam["code"])
        st.markdown(
            f"**Prova ativa:** `{active_exam['code']}` - {active_exam['title']} | Gabarito: {' '.join(active_exam['answer_key'])}"
        )
        if exam_url:
            st.code(exam_url, language="text")
            qr_image = build_qr_image(exam_url)
            st.image(qr_image, caption="QR code da prova ativa", width=220)
        else:
            st.info("Preencha a URL publica do app para gerar o link e o QR.")

        printable_html = generate_printable_template(active_exam, exam_url)
        st.download_button(
            "Baixar modelo HTML desta prova",
            data=printable_html.encode("utf-8"),
            file_name=f"modelo_{active_exam['code'].lower()}.html",
            mime="text/html",
            use_container_width=True,
        )


def render_active_exam_banner():
    active_exam = st.session_state.active_exam
    if not active_exam:
        st.warning("Cadastre ou carregue uma prova antes de corrigir os alunos.")
        return False

    st.info(
        f"Prova ativa: {active_exam['title']} | Codigo: {active_exam['code']} | Gabarito: {' '.join(active_exam['answer_key'])}"
    )
    return True


def main():
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    init_state()

    exams = load_exams()
    sync_exam_from_query(exams)

    st.title("Leitor de Gabarito")
    render_exam_registry()
    st.divider()

    st.subheader("Correcao dos alunos")
    if not render_active_exam_banner():
        render_saved_results()
        return

    nome = st.text_input("Nome do aluno", key="nome_aluno_input")
    turma = st.text_input("Turma", key="turma_input")
    foto = st.camera_input("Tire uma foto do gabarito", key="foto_gabarito_input")

    if foto is None:
        st.info("Tire uma foto do gabarito para iniciar a leitura.")
        render_saved_results()
        return

    imagem = Image.open(foto)
    imagem_bgr = pil_to_bgr(imagem)
    active_exam = st.session_state.active_exam

    try:
        resultado = process_answer_sheet(imagem_bgr)
        respostas = resultado["respostas"]
        acertos, comparacoes = compute_score(respostas, active_exam["answer_key"])
        st.session_state.last_processed = {
            "nome": nome,
            "turma": turma,
            "respostas": respostas,
            "acertos": acertos,
            "comparacoes": comparacoes,
            "prova": active_exam["code"],
        }

        st.subheader("Resultado da leitura")
        st.write("Prova:", active_exam["title"])
        st.write("Codigo da prova:", active_exam["code"])
        st.write("Aluno:", nome or "Nao informado")
        st.write("Turma:", turma or "Nao informada")
        st.write("Respostas lidas:", " ".join(respostas))
        st.write("Gabarito salvo:", " ".join(active_exam["answer_key"]))
        st.write("Acertos:", acertos)
        st.write("Erros:", TOTAL_QUESTIONS - acertos)

        comparison_df = pd.DataFrame(comparacoes)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if st.button("Salvar correcao deste aluno", type="primary", use_container_width=True):
            registro = build_student_record(nome, turma, active_exam, respostas, acertos, comparacoes)
            st.session_state.class_results.append(registro)
            st.success("Correcao salva na turma.")

        render_diagnostics(resultado)
    except ValueError as error:
        st.error(str(error))
        st.image(imagem, caption="Imagem original recebida")

    render_saved_results()


if __name__ == "__main__":
    main()
