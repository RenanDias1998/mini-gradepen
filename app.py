import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(layout="wide")

st.title("📄 Leitor Inteligente de Gabarito")

# -----------------------------
# DADOS DO ALUNO
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

gabarito_texto = st.text_input("Digite o gabarito (ex: A B C D E A B C D E)")

if gabarito_texto:
    gabarito = gabarito_texto.upper().split()
else:
    gabarito = []

# -----------------------------
# CÂMERA
# -----------------------------
foto = st.camera_input("📸 Tire a foto do gabarito")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)

        if 20 < w < 80 and 20 < h < 80:
            bolhas.append((x, y, w, h))

    # ordenar vertical
    bolhas = sorted(bolhas, key=lambda b: (b[1], b[0]))

    respostas = []
    letras = ["A", "B", "C", "D", "E"]

    st.subheader("🔍 Diagnóstico de leitura")

    for i in range(0, len(bolhas), 5):
        grupo = bolhas[i:i+5]

        if len(grupo) < 5:
            continue

        marcacoes = []

        for (x, y, w, h) in grupo:
            area = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(area)
            area_total = w * h

            preenchimento = total / area_total
            marcacoes.append(preenchimento)

        ordenado = sorted(marcacoes, reverse=True)

        if len(ordenado) >= 2:
            confianca = ordenado[0] - ordenado[1]
        else:
            confianca = 0

        indice = np.argmax(marcacoes)

        # REGRA DE DECISÃO (AJUSTÁVEL)
        if marcacoes[indice] > 0.25 and confianca > 0.05:
            resposta = letras[indice]
        else:
            resposta = "?"

        respostas.append(resposta)

        st.write(f"Q{i//5 + 1}: {marcacoes} → {resposta}")

    st.subheader("📌 Respostas detectadas")
    st.write(respostas)

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
            "Erros": ", ".join(erros),
            "Respostas": " ".join(respostas)
        }

        df = pd.DataFrame([dados])

        arquivo = "resultados.csv"

        if os.path.exists(arquivo):
            df_existente = pd.read_csv(arquivo)
            df = pd.concat([df_existente, df], ignore_index=True)

        df.to_csv(arquivo, index=False)

        st.success("📊 Resultado salvo!")

        # -----------------------------
        # MOSTRAR POR TURMA
        # -----------------------------
        st.subheader("📚 Lista por turma")

        tabela = pd.read_csv(arquivo)

        if turma:
            tabela = tabela[tabela["Turma"] == turma]

        st.dataframe(tabela)
