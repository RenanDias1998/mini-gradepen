import streamlit as st
from datetime import datetime
from supabase import create_client

# ==============================
# CONFIG
# ==============================

CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10

# ==============================
# STATE
# ==============================

def init_state():
    defaults = {
        "nav_page": "1. Criar prova",
        "active_exam": None,
        "draft_answer_key": [""] * TOTAL_QUESTIONS,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ==============================
# SAFE NAVIGATION (CORRIGIDO)
# ==============================

def go_to_page(page_name):
    st.session_state["nav_page"] = page_name
    st.rerun()


# ==============================
# SUPABASE
# ==============================

@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ==============================
# DATABASE
# ==============================

def save_exam(exam):
    supabase = get_supabase()

    response = (
        supabase.table("provas")
        .upsert(exam, on_conflict="codigo")
        .execute()
    )

    return response.data[0]


def load_exams():
    supabase = get_supabase()

    response = supabase.table("provas").select("*").execute()

    exams = {}
    for e in response.data:
        exams[e["codigo"]] = e

    return exams


# ==============================
# CREATE EXAM
# ==============================

def render_create_exam_page():
    st.subheader("1. Criar prova")

    titulo = st.text_input("Título da prova")
    data = st.text_input("Data", value=datetime.now().strftime("%Y-%m-%d"))

    if st.button("Criar prova"):
        if not titulo:
            st.warning("Digite um título")
            return

        codigo = titulo.upper().replace(" ", "-")

        exam = {
            "codigo": codigo,
            "titulo": titulo,
            "data_prova": data,
        }

        saved = save_exam(exam)

        st.session_state.active_exam = saved

        st.success("Prova criada!")

        # 🔥 navegação segura
        go_to_page("2. Definir gabarito")


# ==============================
# ANSWER KEY
# ==============================

def render_answer_key_page():
    st.subheader("2. Definir gabarito")

    exams = load_exams()

    if not exams:
        st.info("Nenhuma prova criada ainda")
        return

    codigos = list(exams.keys())

    selected = st.selectbox("Escolha a prova", codigos)

    exam = exams[selected]
    st.session_state.active_exam = exam

    for i in range(TOTAL_QUESTIONS):
        col1, col2, col3, col4, col5 = st.columns(5)

        for idx, choice in enumerate(CHOICES):
            if st.button(choice, key=f"{i}_{choice}"):
                st.session_state.draft_answer_key[i] = choice

    if st.button("Salvar gabarito"):
        if "" in st.session_state.draft_answer_key:
            st.warning("Preencha tudo")
            return

        exam["gabarito"] = st.session_state.draft_answer_key

        save_exam(exam)

        st.success("Gabarito salvo!")

        go_to_page("3. Corrigir gabaritos")


# ==============================
# CORRECTION
# ==============================

def render_correction_page():
    st.subheader("3. Corrigir")

    exam = st.session_state.active_exam

    if not exam:
        st.warning("Nenhuma prova ativa")
        return

    nome = st.text_input("Nome do aluno")

    st.write("Simulação de correção (sem visão computacional ainda)")

    respostas = st.text_input("Digite respostas (ex: A B C D E...)")

    if st.button("Corrigir"):
        gabarito = exam.get("gabarito", [])

        respostas = respostas.split()

        acertos = 0

        for i in range(min(len(respostas), len(gabarito))):
            if respostas[i] == gabarito[i]:
                acertos += 1

        st.success(f"Acertos: {acertos}")


# ==============================
# MAIN
# ==============================

def main():
    st.set_page_config(page_title="Leitor de Gabarito")

    init_state()

    st.title("Leitor de Gabarito")

    page = st.sidebar.radio(
        "Etapas",
        ["1. Criar prova", "2. Definir gabarito", "3. Corrigir gabaritos"],
        index=["1. Criar prova", "2. Definir gabarito", "3. Corrigir gabaritos"].index(st.session_state.nav_page)
    )

    st.session_state.nav_page = page

    if page == "1. Criar prova":
        render_create_exam_page()
    elif page == "2. Definir gabarito":
        render_answer_key_page()
    else:
        render_correction_page()


if __name__ == "__main__":
    main()
