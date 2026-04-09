import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.title("📄 Leitor Inteligente de Gabarito")

# -----------------------------
# DADOS DO ALUNO
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

gabarito_texto = st.text_input("Digite o gabarito (ex: C D B D B C B D D A)")

if gabarito_texto:
    gabarito = gabarito_texto.upper().split()
else:
    gabarito = []

# -----------------------------
# CÂMERA DO CELULAR
# -----------------------------
foto = st.camera_input("📸 Tire a foto do gabarito")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # melhorar contraste
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        # tamanho das bolinhas
        if 20 < w < 80 and 20 < h < 80:
            bolhas.append((x, y, w, h))

    # ordenar de cima pra baixo (vertical)
    bolhas = sorted(bolhas, key=lambda b: (b[1], b[0]))

    respostas = []
    letras = ["A", "B", "C", "D", "E"]

    # agrupar de 5 em 5 (cada questão)
    for i in range(0, len(bolhas), 5):
        grupo = bolhas[i:i+5]

        if len(grupo) < 5:
            continue

        marcacoes = []

        for (x, y, w, h) in grupo:
            area = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(area)
            marcacoes.append(total)

        resposta = letras[np.argmax(marcacoes)]
        respostas.append(resposta)

    st.write("📌 Respostas detectadas:", respostas)

    # -----------------------------
    # CORREÇÃO
    # -----------------------------
    if gabarito:

        acertos = 0
        erros = []

        for i in range(min(len(gabarito), len(respostas))):
            if respostas[i] == gabarito[i]:
                acertos += 1
            else:
                erros.append(f"Q{i+1}")

        st.success(f"✅ Acertos: {acertos}")
        st.error(f"❌ Erros: {erros}")

        # -----------------------------
        # SALVAR RESULTADOS
        # -----------------------------
        dados = {
            "Nome": nome,
            "Turma": turma,
            "Acertos": acertos,
            "Erros": ", ".join(erros)
        }

        df = pd.DataFrame([dados])

        arquivo = "resultados.csv"

        if os.path.exists(arquivo):
            df_existente = pd.read_csv(arquivo)
            df = pd.concat([df_existente, df], ignore_index=True)

        df.to_csv(arquivo, index=False)

        st.success("📊 Resultado salvo!")

        # mostrar lista por turma
        st.subheader("📚 Lista de resultados")

        tabela = pd.read_csv(arquivo)

        if turma:
            tabela = tabela[tabela["Turma"] == turma]

        st.dataframe(tabela)
