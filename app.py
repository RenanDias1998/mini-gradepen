import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(layout="wide")
st.title("📄 Leitor Inteligente de Gabarito")

st.info("📸 Tire a foto de cima, com a folha inteira visível")

# -----------------------------
# DADOS
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")
gabarito_texto = st.text_input("Gabarito (ex: A B C D E A B C D E)")

gabarito = gabarito_texto.upper().split() if gabarito_texto else []

# -----------------------------
# FUNÇÃO: ORDENAR PONTOS
# -----------------------------
def ordenar_pontos(pts):
    pts = pts.reshape(4, 2)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(soma)],
        pts[np.argmin(diff)],
        pts[np.argmax(soma)],
        pts[np.argmax(diff)]
    ], dtype="float32")

# -----------------------------
# FUNÇÃO: ALINHAR FOLHA
# -----------------------------
def alinhar_imagem(imagem):

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for c in contornos:
        peri = cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(aprox) == 4:
            pontos = ordenar_pontos(aprox)

            (tl, tr, br, bl) = pontos

            largura = int(max(
                np.linalg.norm(br - bl),
                np.linalg.norm(tr - tl)
            ))

            altura = int(max(
                np.linalg.norm(tr - br),
                np.linalg.norm(tl - bl)
            ))

            destino = np.array([
                [0, 0],
                [largura-1, 0],
                [largura-1, altura-1],
                [0, altura-1]
            ], dtype="float32")

            matriz = cv2.getPerspectiveTransform(pontos, destino)
            warp = cv2.warpPerspective(imagem, matriz, (largura, altura))

            return warp

    return imagem

# -----------------------------
# CÂMERA
# -----------------------------
foto = st.camera_input("📸 Tirar foto")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    # alinhar automaticamente
    imagem = alinhar_imagem(imagem)

    st.image(imagem, caption="Imagem alinhada", use_column_width=True)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    altura, largura = thresh.shape

    # -----------------------------
    # RECORTE DO GABARITO (lado direito)
    # -----------------------------
    x1 = int(largura * 0.55)
    x2 = int(largura * 0.95)

    y1 = int(altura * 0.15)
    y2 = int(altura * 0.90)

    gabarito_img = thresh[y1:y2, x1:x2]

    st.image(gabarito_img, caption="Área do gabarito", use_column_width=True)

    # -----------------------------
    # LEITURA
    # -----------------------------
    respostas = []
    letras = ["A", "B", "C", "D", "E"]

    num_questoes = len(gabarito) if gabarito else 10
    colunas = 5

    h, w = gabarito_img.shape

    altura_bloco = h // num_questoes
    largura_bloco = w // colunas

    st.subheader("🔍 Diagnóstico")

    for i in range(num_questoes):

        valores = []

        for j in range(colunas):

            yi1 = i * altura_bloco
            yi2 = (i + 1) * altura_bloco

            xi1 = j * largura_bloco
            xi2 = (j + 1) * largura_bloco

            bloco = gabarito_img[yi1:yi2, xi1:xi2]

            total = cv2.countNonZero(bloco)
            area_total = bloco.size

            porcentagem = total / area_total
            valores.append(porcentagem)

        maior = max(valores)
        segundo = sorted(valores, reverse=True)[1]

        indice = valores.index(maior)

        # ajuste fino para caneta
        if maior > 0.20 and (maior - segundo) > 0.05:
            resposta = letras[indice]
        else:
            resposta = "?"

        respostas.append(resposta)

        st.write(f"Q{i+1}: {['%.2f' % v for v in valores]} → {resposta}")

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

        # salvar
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
# LISTA
# -----------------------------
st.subheader("📚 Resultados")

arquivo = "resultados.csv"

if os.path.exists(arquivo):
    df = pd.read_csv(arquivo)

    if len(df) > 0:
        st.dataframe(df)
    else:
        st.warning("Sem dados ainda")
