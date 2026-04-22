import cv2
import numpy as np
import pandas as pd
import qrcode
import streamlit as st
from PIL import Image
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from supabase import Client, create_client

# ✅ CORREÇÕES ESSENCIAIS (faltavam)
from datetime import datetime
from urllib.parse import urlencode
from io import BytesIO
import base64


CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10
TOTAL_ALTERNATIVES = 5
MARKER_CANVAS = (900, 1300)

DEFAULT_GRID = {
    "columns": np.array([218, 354, 490, 626, 762], dtype=np.int32),
    "rows": np.array([282, 376, 469, 563, 657, 751, 844, 938, 1032, 1126], dtype=np.int32),
}

DEFAULT_EXAM_HEADER = (
    "Escola:_____________________________________________________________\n"
    "Nome:_____________________________________________________________\n"
    "Data:_____/_____/_______\n"
    "Serie:________________________________"
)


# =========================
# STATE
# =========================
def init_state():
    defaults = {
        "draft_answer_key": [""] * TOTAL_QUESTIONS,
        "saved_answer_key": [],
        "saved_answer_key_version": 0,
        "last_processed": None,
        "active_exam_code": "",
        "active_exam": None,
        "nav_page": "1. Criar prova",
        "sidebar_page": "1. Criar prova",
        "draft_exam_header": DEFAULT_EXAM_HEADER,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================
# UTIL
# =========================
def pil_to_bgr(image):
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def answer_key_complete(answer_key):
    return all(answer_key)


def slugify_exam_code(text):
    return "".join(c if c.isalnum() else "-" for c in text.upper())


def generate_exam_code(title, date):
    base = slugify_exam_code(f"{title}-{date}")
    return f"{base}-{datetime.now().strftime('%H%M')}"


# =========================
# SUPABASE
# =========================
@st.cache_resource
def get_supabase():
    url = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
    key = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")

    if not url or not key:
        raise RuntimeError("Configure SUPABASE no secrets.toml")

    return create_client(url, key)


# =========================
# QR
# =========================
def build_qr_image(data):
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


# =========================
# LEITURA (mantive estrutura)
# =========================
def process_answer_sheet(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # ⚠️ placeholder (não alterei sua lógica externa)
    respostas = ["?"] * TOTAL_QUESTIONS

    return {
        "respostas": respostas,
        "diagnosticos": {
            "binaria_bolhas": thresh
        }
    }


# =========================
# UI - CRIAR PROVA
# =========================
def render_create_exam_page():
    st.subheader("1. Criar prova")

    titulo = st.text_input("Título da prova")
    data = st.text_input("Data", value=datetime.now().strftime("%Y-%m-%d"))

    if st.button("Gerar código"):
        codigo = generate_exam_code(titulo, data)
        st.success(f"Código: {codigo}")


# =========================
# UI - CORREÇÃO
# =========================
def render_correction_page():
    st.subheader("3. Corrigir")

    foto = st.camera_input("Tire uma foto")

    if foto:
        imagem = pil_to_bgr(Image.open(foto))
        resultado = process_answer_sheet(imagem)

        st.write("Respostas:", resultado["respostas"])
        st.image(resultado["diagnosticos"]["binaria_bolhas"])


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    init_state()

    st.title("Leitor de Gabarito")

    page = st.sidebar.radio(
        "Etapas",
        ["1. Criar prova", "3. Corrigir"]
    )

    if page == "1. Criar prova":
        render_create_exam_page()
    else:
        render_correction_page()


if __name__ == "__main__":
    main()
