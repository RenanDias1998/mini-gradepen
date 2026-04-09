import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("📄 Leitor de Gabarito")

# Nome do aluno
nome = st.text_input("Nome do aluno")

# Gabarito
gabarito_texto = st.text_input("Digite o gabarito (ex: C D B D B C B D D A)")

if gabarito_texto:
    gabarito = gabarito_texto.upper().split()
else:
    gabarito = []

# Upload da imagem
foto = st.file_uploader("📸 Envie a foto do gabarito", type=["jpg", "png"])

if foto is not None:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        if 20 < w < 80 and 20 < h < 80:
            bolhas.append((x, y, w, h))

    bolhas = sorted(bolhas, key=lambda b: (b[1], b[0]))

    respostas_letras = []
    letras = ["A", "B", "C", "D", "E"]

    for i in range(0, len(bolhas), 5):
        grupo = bolhas[i:i+5]

        marcacoes = []

        for (x, y, w, h) in grupo:
            area = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(area)
            marcacoes.append(total)

        if marcacoes:
            resposta = letras[np.argmax(marcacoes)]
            respostas_letras.append(resposta)

    st.write("Respostas detectadas:", respostas_letras)

    if gabarito:
        acertos = 0
        erros = []

        for i in range(min(len(gabarito), len(respostas_letras))):
            if respostas_letras[i] == gabarito[i]:
                acertos += 1
            else:
                erros.append(f"Q{i+1}")

        st.write("✅ Acertos:", acertos)
        st.write("❌ Erros:", erros)
