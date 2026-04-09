import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(layout="wide")

st.title("📄 Leitor Inteligente de Gabarito")

st.info("📸 Tire a foto de cima (90°), com boa luz e distância de aproximadamente 30cm")

# -----------------------------
# DADOS
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

gabarito_texto = st.text_input("Gabarito (ex: A B C D E A B C D E)")

gabarito = gabarito_texto.upper().split() if gabarito_texto else []

# -----------------------------
# CAMERA
# -----------------------------
foto = st.camera_input("Tirar foto")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # melhora MUITO a leitura
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []

    for c in contornos:
        area = cv2.contourArea(c)

        if 500 < area < 4000:  # filtro mais preciso
            x, y, w, h = cv2.boundingRect(c)
            bolhas.append((x, y, w, h))

    # ordenar vertical
    bolhas = sorted(bolhas, key=lambda b: (b[1], b[0]))

    respostas = []
    letras = ["A", "B", "C", "D", "E"]

    st.subheader("🔍 Diagnóstico")

    for i in range(0, len(bolhas), 5):
        grupo = bolhas[i:i+5]

        if len(grupo) < 5:
            continue

        valores = []

        for (x, y, w, h) in grupo:
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            area_total = w * h

            porcentagem = total / area_total
            valores.append(porcentagem)

        # ordenação para comparar
        maior = max(valores)
        segundo = sorted(valores, reverse=True)[1]

        indice = valores.index(maior)

        # 🔥 NOVA REGRA (MUITO MAIS PRECISA)
        if maior > 0.35 and (maior - segundo) > 0.10:
            resposta = letras[indice]
        else:
            resposta = "?"

        respostas.append(resposta)

        st.write(f"Q{i//5 + 1}: {['%.2f' % v for v in valores]} → {resposta}")

    st.subheader("📌 Respostas")
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
        # SALVAR
        # -----------------------------
        dados = {
            "Nome": nome,
            "Turma": turma,
            "Acertos": acertos,
            "Erros": ", ".join(erros)
        }

        arquivo = "resultados.csv"

        df_novo = pd.DataFrame([dados])

        if os.path.exists(arquivo):
            df = pd.read_csv(arquivo)
            df = pd.concat([df, df_novo], ignore_index=True)
        else:
            df = df_novo

        df.to_csv(arquivo, index=False)

        st.success("💾 Salvo!")

# -----------------------------
# LISTA + EXCLUIR
# -----------------------------
st.subheader("📚 Resultados")

arquivo = "resultados.csv"

if os.path.exists(arquivo):
    df = pd.read_csv(arquivo)

    st.dataframe(df)

    indice = st.number_input("Digite o número da linha para excluir", min_value=0, max_value=len(df)-1, step=1)

    if st.button("🗑️ Excluir linha"):
        df = df.drop(indice)
        df.to_csv(arquivo, index=False)
        st.success("Linha excluída! Atualize a página.")
