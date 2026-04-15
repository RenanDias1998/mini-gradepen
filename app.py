def main():
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    st.title("Leitor de Gabarito")

    # Inputs com KEY (ESSENCIAL)
    nome = st.text_input("Nome do aluno", key="nome_aluno")
    turma = st.text_input("Turma", key="turma_aluno")
    foto = st.camera_input("Tire uma foto do gabarito", key="camera_gabarito")

    # Só continua se tiver foto
    if foto is None:
        st.info("📸 Tire uma foto do gabarito para começar.")
        return

    # Feedback visual
    st.success("Imagem capturada com sucesso!")

    imagem = Image.open(foto)
    imagem_bgr = pil_to_bgr(imagem)

    # Botão para processar (evita rerun automático)
    if st.button("🔍 Processar gabarito", key="processar_btn"):

        with st.spinner("Processando imagem..."):
            try:
                resultado = process_answer_sheet(imagem_bgr)
                respostas = resultado["respostas"]

                acertos = sum(
                    1 for resposta, gabarito in zip(respostas, ANSWER_KEY)
                    if resposta == gabarito
                )

                st.subheader("📊 Resultado")
                st.write("👤 Aluno:", nome or "Nao informado")
                st.write("🏫 Turma:", turma or "Nao informada")
                st.write("📝 Respostas lidas:", " ".join(respostas))
                st.write("✅ Acertos:", acertos)

                render_diagnostics(resultado)

            except ValueError as error:
                st.error(str(error))
                st.image(imagem, caption="Imagem original recebida")
