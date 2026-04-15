import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Leitor de Gabarito")

nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

foto = st.camera_input("Tire uma foto do gabarito")

if foto is not None:
    imagem = Image.open(foto)
    imagem_np = np.array(imagem)

    cinza = cv2.cvtColor(imagem_np, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cinza, 150, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if 20 < w < 60 and 20 < h < 60:
            bolhas.append((x, y, w, h))

    bolhas = sorted(bolhas, key=lambda b: (b[1], b[0]))

    respostas = []

    for i in range(0, len(bolhas), 5):
        grupo = bolhas[i:i+5]

        if len(grupo) < 5:
            continue

        marcacao = []

        for (x, y, w, h) in grupo:
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            marcacao.append(total)

        resposta = np.argmax(marcacao)
        respostas.append(resposta)

    letras = ["A", "B", "C", "D", "E"]
    respostas_letras = [letras[i] for i in respostas]

    gabarito = ["C","D","B","D","B","C","B","D","D","A"]

    acertos = 0

    for i in range(min(len(gabarito), len(respostas_letras))):
        if respostas_letras[i] == gabarito[i]:
            acertos += 1

    st.write("Aluno:", nome)
    st.write("Turma:", turma)
    st.write("Acertos:", acertos)
import cv2
import numpy as np
import streamlit as st
from PIL import Image


ANSWER_KEY = ["C", "D", "B", "D", "B", "C", "B", "D", "D", "A"]
CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10
TOTAL_ALTERNATIVES = 5
MARKER_CANVAS = (900, 1300)
DEFAULT_GRID = {
    "columns": np.array([218, 354, 490, 626, 762], dtype=np.int32),
    "rows": np.array([282, 376, 469, 563, 657, 751, 844, 938, 1032, 1126], dtype=np.int32),
}


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

    for question_index, center_y in enumerate(rows):
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


def main():
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    st.title("Leitor de Gabarito")

    nome = st.text_input("Nome do aluno")
    turma = st.text_input("Turma")
    foto = st.camera_input("Tire uma foto do gabarito")

    if foto is None:
        return

    imagem = Image.open(foto)
    imagem_bgr = pil_to_bgr(imagem)

    try:
        resultado = process_answer_sheet(imagem_bgr)
        respostas = resultado["respostas"]
        acertos = sum(
            1 for resposta, gabarito in zip(respostas, ANSWER_KEY) if resposta == gabarito
        )

        st.subheader("Resultado")
        st.write("Aluno:", nome or "Nao informado")
        st.write("Turma:", turma or "Nao informada")
        st.write("Respostas lidas:", " ".join(respostas))
        st.write("Acertos:", acertos)

        render_diagnostics(resultado)
    except ValueError as error:
        st.error(str(error))
        st.image(imagem, caption="Imagem original recebida")


if __name__ == "__main__":
    main()
