from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10
WARP_W = 1400
WARP_H = 2000
PANEL_X1 = 700
PANEL_Y1 = 420
PANEL_X2 = 1380
PANEL_Y2 = 1820


@dataclass
class OMRQuestionResult:
    question: int
    selected: str
    confidence: float
    status: str
    densities: List[float]


@dataclass
class OMRScanResult:
    answers: List[str]
    questions: List[OMRQuestionResult]
    page_view: np.ndarray
    panel_view: np.ndarray
    binary_panel: np.ndarray


class OMRException(Exception):
    pass


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def find_page_warp(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise OMRException("Nao foi possivel detectar a folha.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    page = None
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            page = approx.reshape(4, 2)
            break

    if page is None:
        raise OMRException("Nao foi possivel alinhar a folha. Tente foto frontal e boa luz.")

    src = order_corners(page)
    dst = np.array(
        [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image_bgr, matrix, (WARP_W, WARP_H))


def find_marker_centers(panel_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    min_area = int((h * w) * 0.001)
    max_area = int((h * w) * 0.02)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        ratio = cw / float(ch)
        if ratio < 0.7 or ratio > 1.3:
            continue

        fill = area / float(cw * ch)
        if fill < 0.6:
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0
        candidates.append((cx, cy, area))

    if len(candidates) < 4:
        raise OMRException("Marcadores pretos do painel nao detectados. Verifique contraste e enquadramento.")

    pts = np.array([[c[0], c[1]] for c in sorted(candidates, key=lambda t: t[2], reverse=True)[:8]], dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def rectify_answer_panel(page_warp: np.ndarray) -> np.ndarray:
    rough = page_warp[PANEL_Y1:PANEL_Y2, PANEL_X1:PANEL_X2].copy()
    corners = find_marker_centers(rough)

    out_w, out_h = 800, 1500
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(order_corners(corners), dst)
    return cv2.warpPerspective(rough, matrix, (out_w, out_h))


def normalize_panel_without_markers(page_warp: np.ndarray) -> np.ndarray:
    rough = page_warp[PANEL_Y1:PANEL_Y2, PANEL_X1:PANEL_X2].copy()
    return cv2.resize(rough, (800, 1500), interpolation=cv2.INTER_LINEAR)


def read_answers_from_panel(panel_bgr: np.ndarray, total_questions: int = TOTAL_QUESTIONS):
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    left = int(w * 0.23)
    right = int(w * 0.93)
    top = int(h * 0.22)
    bottom = int(h * 0.90)

    q_height = (bottom - top) / float(total_questions)
    c_width = (right - left) / float(len(CHOICES))

    answers = []
    results = []

    for q_idx in range(total_questions):
        y1 = int(top + q_idx * q_height)
        y2 = int(top + (q_idx + 1) * q_height)
        row_scores = []

        for c_idx in range(len(CHOICES)):
            x1 = int(left + c_idx * c_width)
            x2 = int(left + (c_idx + 1) * c_width)

            roi = binary[y1:y2, x1:x2]
            if roi.size == 0:
                row_scores.append(0.0)
                continue

            margin_y = max(1, int(roi.shape[0] * 0.20))
            margin_x = max(1, int(roi.shape[1] * 0.20))
            core = roi[margin_y:-margin_y, margin_x:-margin_x]
            if core.size == 0:
                row_scores.append(0.0)
                continue

            density = cv2.countNonZero(core) / float(core.size)
            row_scores.append(float(density))

        sorted_indices = np.argsort(row_scores)[::-1]
        best_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1])

        best = row_scores[best_idx]
        second = row_scores[second_idx]
        confidence = float(best - second)

        if best < 0.09:
            selected = "-"
            status = "em_branco"
        elif confidence < 0.012:
            selected = "*"
            status = "duvida"
        else:
            selected = CHOICES[best_idx]
            status = "ok"

        answers.append(selected)
        results.append(
            OMRQuestionResult(
                question=q_idx + 1,
                selected=selected,
                confidence=confidence,
                status=status,
                densities=row_scores,
            )
        )

    return answers, results, binary


def run_omr(image_bgr: np.ndarray) -> OMRScanResult:
    page = find_page_warp(image_bgr)
    try:
        panel = rectify_answer_panel(page)
    except OMRException:
        panel = normalize_panel_without_markers(page)
    answers, question_results, binary_panel = read_answers_from_panel(panel)

    return OMRScanResult(
        answers=answers,
        questions=question_results,
        page_view=page,
        panel_view=panel,
        binary_panel=binary_panel,
    )
