import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(layout="wide")

st.title("📄 Leitor Inteligente de Gabarito (Modelo Oficial)")

st.info("📸 Tire a foto incluindo os quadrados pretos e o gabarito completo")

# -----------------------------
# DADOS
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

gabarito_texto = st.text_input("Gabarito (ex: C D B D B C B D D A)")
gabarito = gabarito_texto.upper().split() if gabarito_texto else []

# -----------------------------
# CAMERA
# -----------------------------
foto = st.camera_input("📸 Tirar foto")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # -----------------------------
    # DETECTAR QUADRADOS PRETOS
    # -----------------------------
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quadrados = []

    for c in contornos:
        area = cv2.contourArea(c)

        if area > 2000:
            x, y, w, h = cv2.boundingRect(c)

            proporcao = w / float(h)

            if 0.8 < proporcao < 1.2:
                quadrados.append((x, y, w, h))

    if len(quadrados) >= 4:

        # ordenar
        quadrados = sorted(quadrados, key=lambda q: (q[1], q[0]))

        # pegar região do gabarito (lado direito)
        xs = [q[0] for q in quadrados]
        ys = [q[1] for q in quadrados]
        ws = [q[2] for q in quadrados]
        hs = [q[3] for q in quadrados]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max([xs[i] + ws[i] for i in range(len(xs))])
        y_max = max([ys[i] + hs[i] for i in range(len(ys))])

        # recorte geral
        recorte = thresh[y_min:y_max, x_min:x_max]

        # cortar só o lado direito (gabarito)
        altura, largura = recorte.shape
        recorte = recorte[:, int(largura*0.45):]

        st.image(recorte, caption="Área do gabarito detectada")

        # -----------------------------
        # LEITURA POR GRADE
        # -----------------------------
        respostas = []
        letras = ["A", "B", "C", "D", "E"]

        num_questoes = len(gabarito) if gabarito else 10
        colunas = 5

        altura_bloco = recorte.shape[0] // num_questoes
        largura_bloco = recorte.shape[1] // colunas

        st.subheader("🔍 Diagnóstico")

        for i in range(num_questoes):

            valores = []

            for j in range(colunas):

                y1 = i * altura_bloco
                y2 = (i + 1) * altura_bloco

                x1 = j * largura_bloco
                x2 = (j + 1) * largura_bloco

                bloco = recorte[y1:y2, x1:x2]

                total = cv2.countNonZero(bloco)
                area_total = bloco.size

                porcentagem = total / area_total
                valores.append(porcentagem)

            maior = max(valores)
            segundo = sorted(valores, reverse=True)[1]

            indice = valores.index(maior)

            if maior > 0.35 and (maior - segundo) > 0.08:
                resposta = letras[indice]
            else:
                resposta = "?"

            respostas.append(resposta)

            st.write(f"Q{i+1}: {['%.2f' % v for v in valores]} → {resposta}")

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

            st.success("💾 Resultado salvo!")

    else:
        st.error("❌ Não foi possível identificar os marcadores (quadrados pretos)")

# -----------------------------
# LISTA DE TURMA
# -----------------------------
st.subheader("📚 Lista da turma")

arquivo = "resultados.csv"

if os.path.exists(arquivo):
    df = pd.read_csv(arquivo)

    if turma:
        df = df[df["Turma"] == turma]

    st.dataframe(df)
