from io import BytesIO

import cv2
import numpy as np
import pandas as pd
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


def init_state():
    if "draft_answer_key" not in st.session_state:
        st.session_state.draft_answer_key = [""] * TOTAL_QUESTIONS
    if "saved_answer_key" not in st.session_state:
        st.session_state.saved_answer_key = []
    if "saved_answer_key_version" not in st.session_state:
        st.session_state.saved_answer_key_version = 0
    if "class_results" not in st.session_state:
        st.session_state.class_results = []
    if "last_processed" not in st.session_state:
        st.session_state.last_processed = None


def pil_to_bgr(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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
    search[:, int(width * 0.48) :] = dark[:, int(width * 0.48) :]

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

    col_window = binary[int(height * 0.18) : int(height * 0.88), int(width * 0.15) : int(width * 0.92)]
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

    row_window = binary[int(height * 0.18) : int(height * 0.92), int(width * 0.18) : int(width * 0.86)]
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


def answer_key_complete(answer_key):
    return len(answer_key) == TOTAL_QUESTIONS and all(answer_key)


def render_answer_key_editor():
    st.subheader("Gabarito do professor")
    st.caption("Clique na alternativa de cada questao. A escolha permanece ate voce salvar.")

    for question_index in range(TOTAL_QUESTIONS):
        current_choice = st.session_state.draft_answer_key[question_index]
        row_columns = st.columns([1.2, 1, 1, 1, 1, 1, 1.6])
        row_columns[0].markdown(f"**Q{question_index + 1}**")

        for choice_index, choice in enumerate(CHOICES, start=1):
            selected = current_choice == choice
            label = f"{choice} {'🟩' if selected else ''}".strip()
            button_type = "primary" if selected else "secondary"
            row_columns[choice_index].button(
                label,
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

    col1, col2, col3 = st.columns([1.2, 1.2, 3])
    if col1.button("Salvar gabarito", type="primary", use_container_width=True):
        if answer_key_complete(st.session_state.draft_answer_key):
            st.session_state.saved_answer_key = st.session_state.draft_answer_key.copy()
            st.session_state.saved_answer_key_version += 1
            st.success("Gabarito salvo. Ele sera usado ate voce salvar outro.")
        else:
            st.warning("Preencha todas as questoes antes de salvar o gabarito.")

    if col2.button("Limpar selecoes", use_container_width=True):
        st.session_state.draft_answer_key = [""] * TOTAL_QUESTIONS
        st.info("Selecoes do gabarito limpas.")

    saved_answer_key = st.session_state.saved_answer_key
    if answer_key_complete(saved_answer_key):
        col3.markdown(
            f"**Gabarito ativo:** <span style='color:#15803d; font-weight:700;'>{' '.join(saved_answer_key)}</span>",
            unsafe_allow_html=True,
        )
    else:
        col3.warning("Nenhum gabarito salvo ainda.")


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


def build_student_record(name, class_name, student_answers, correct_answers, comparisons):
    record = {
        "Aluno": name or "Nao informado",
        "Turma": class_name or "Nao informada",
        "Acertos": correct_answers,
        "Erros": TOTAL_QUESTIONS - correct_answers,
        "Respostas Lidas": " ".join(student_answers),
        "Gabarito Usado": " ".join(st.session_state.saved_answer_key),
        "Versao Gabarito": st.session_state.saved_answer_key_version,
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


def main():
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    init_state()

    st.title("Leitor de Gabarito")
    render_answer_key_editor()
    st.divider()

    st.subheader("Correcao dos alunos")
    nome = st.text_input("Nome do aluno", key="nome_aluno_input")
    turma = st.text_input("Turma", key="turma_input")
    foto = st.camera_input("Tire uma foto do gabarito", key="foto_gabarito_input")

    saved_answer_key = st.session_state.saved_answer_key
    if not answer_key_complete(saved_answer_key):
        st.warning("Salve o gabarito do professor antes de corrigir a turma.")
        render_saved_results()
        return

    if foto is None:
        st.info("Tire uma foto do gabarito para iniciar a leitura.")
        render_saved_results()
        return

    imagem = Image.open(foto)
    imagem_bgr = pil_to_bgr(imagem)

    try:
        resultado = process_answer_sheet(imagem_bgr)
        respostas = resultado["respostas"]
        acertos, comparacoes = compute_score(respostas, saved_answer_key)
        st.session_state.last_processed = {
            "nome": nome,
            "turma": turma,
            "respostas": respostas,
            "acertos": acertos,
            "comparacoes": comparacoes,
        }

        st.subheader("Resultado da leitura")
        st.write("Aluno:", nome or "Nao informado")
        st.write("Turma:", turma or "Nao informada")
        st.write("Respostas lidas:", " ".join(respostas))
        st.write("Gabarito salvo:", " ".join(saved_answer_key))
        st.write("Acertos:", acertos)
        st.write("Erros:", TOTAL_QUESTIONS - acertos)

        comparison_df = pd.DataFrame(comparacoes)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        if st.button("Salvar correcao deste aluno", type="primary", use_container_width=True):
            registro = build_student_record(nome, turma, respostas, acertos, comparacoes)
            st.session_state.class_results.append(registro)
            st.success("Correcao salva na turma. Ela permanecera ate voce baixar o Excel ou limpar os dados.")

        render_diagnostics(resultado)
    except ValueError as error:
        st.error(str(error))
        st.image(imagem, caption="Imagem original recebida")

    render_saved_results()


if __name__ == "__main__":
    main()
