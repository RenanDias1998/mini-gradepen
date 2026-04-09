import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os

st.set_page_config(layout="wide")

st.title("📄 Leitor Inteligente de Gabarito")

st.info("📸 Tire a foto de cima (90°), com boa iluminação e distância aproximada de 30cm")

# -----------------------------
# DADOS DO ALUNO
# -----------------------------
nome = st.text_input("Nome do aluno")
turma = st.text_input("Turma")

gabarito_texto = st.text_input("Digite o gabarito (ex: A B C D E A B C D E)")

gabarito = gabarito_texto.upper().split() if gabarito_texto else []

# -----------------------------
# CÂMERA
# -----------------------------
foto = st.camera_input("📸 Tirar foto do gabarito")

if foto is not None and nome and turma:

    imagem = Image.open(foto)
    imagem = np.array(imagem)

    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # melhora contraste e reduz erro de iluminação
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    altura, largura = thresh.shape

    respostas = []
    letras = ["A", "B", "C", "D", "E"]

    num_questoes = len(gabarito) if gabarito else 10
    colunas = 5

    altura_bloco = altura // num_questoes
    largura_bloco = largura // colunas

    st.subheader("🔍 Diagnóstico da leitura")

    for i in range(num_questoes):

        valores = []

        for j in range(colunas):

            y1 = i * altura_bloco
            y2 = (i + 1) * altura_bloco

            x1 = j * largura_bloco
            x2 = (j + 1) * largura_bloco

            bloco = thresh[y1:y2, x1:x2]

            total = cv2.countNonZero(bloco)
            area_total = bloco.size

            porcentagem = total / area_total
            valores.append(porcentagem)

        maior = max(valores)
        segundo = sorted(valores, reverse=True)[1]

        indice = valores.index(maior)

        # REGRA DE DECISÃO (AJUSTÁVEL)
        if maior > 0.30 and (maior - segundo) > 0.08:
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

        arquivo = "resultados.csv"

        df_novo = pd.DataFrame([dados])

        if os.path.exists(arquivo):
            df = pd.read_csv(arquivo)
            df = pd.concat([df, df_novo], ignore_index=True)
        else:
            df = df_novo

        df.to_csv(arquivo, index=False)

        st.success("💾 Resultado salvo!")

# -----------------------------
# LISTA + EXCLUSÃO
# -----------------------------
st.subheader("📚 Resultados")

arquivo = "resultados.csv"

if os.path.exists(arquivo):
    df = pd.read_csv(arquivo)

    if len(df) > 0:
        st.dataframe(df)

        indice = st.number_input(
            "Digite o número da linha para excluir",
            min_value=0,
            max_value=len(df)-1,
            step=1
        )

        if st.button("🗑️ Excluir linha"):
            df = df.drop(indice).reset_index(drop=True)
            df.to_csv(arquivo, index=False)
            st.success("Linha excluída! Atualize a página.")
    else:
        st.warning("Nenhum resultado salvo ainda.")
