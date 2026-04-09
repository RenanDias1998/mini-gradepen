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