import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client

from omr_core import OMRException, OMRScanResult, run_omr

CHOICES = ["A", "B", "C", "D", "E"]
TOTAL_QUESTIONS = 10


def init_state() -> None:
    defaults = {
        "nav_page": "1. Criar prova",
        "active_exam": None,
        "draft_answer_key": [""] * TOTAL_QUESTIONS,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_to_page(page_name: str) -> None:
    st.session_state["nav_page"] = page_name
    st.rerun()


def normalize_exam_code(title: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9\s-]", "", title).strip().upper()
    return re.sub(r"\s+", "-", clean)


def parse_exam_date(raw: str) -> Optional[str]:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


@st.cache_resource
def get_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        raise OMRException("Credenciais do Supabase nao configuradas em st.secrets.")
    return create_client(url, key)


def save_exam(exam: Dict) -> Dict:
    try:
        supabase = get_supabase()
        response = supabase.table("provas").upsert(exam, on_conflict="codigo").execute()
        if not response.data:
            raise OMRException("Falha ao salvar prova no banco.")
        return response.data[0]
    except Exception as exc:
        raise OMRException(f"Erro ao salvar prova: {exc}") from exc


def load_exams() -> Dict[str, Dict]:
    try:
        supabase = get_supabase()
        response = supabase.table("provas").select("*").execute()
        exams = {e["codigo"]: e for e in response.data or []}
        return exams
    except Exception as exc:
        raise OMRException(f"Erro ao carregar provas: {exc}") from exc


def render_create_exam_page() -> None:
    st.subheader("1. Criar prova")

    titulo = st.text_input("Titulo da prova")
    data_raw = st.text_input("Data (AAAA-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))

    if st.button("Criar prova"):
        if not titulo.strip():
            st.warning("Digite um titulo.")
            return

        data = parse_exam_date(data_raw.strip())
        if not data:
            st.warning("Data invalida. Use formato AAAA-MM-DD.")
            return

        codigo = normalize_exam_code(titulo)
        if not codigo:
            st.warning("Nao foi possivel gerar codigo da prova.")
            return

        exam = {"codigo": codigo, "titulo": titulo.strip(), "data_prova": data}

        try:
            saved = save_exam(exam)
        except OMRException as exc:
            st.error(str(exc))
            return

        st.session_state.active_exam = saved
        st.session_state.draft_answer_key = [""] * TOTAL_QUESTIONS

        st.success(f"Prova criada com codigo: {codigo}")
        go_to_page("2. Definir gabarito")


def render_answer_key_page() -> None:
    st.subheader("2. Definir gabarito")

    try:
        exams = load_exams()
    except OMRException as exc:
        st.error(str(exc))
        return

    if not exams:
        st.info("Nenhuma prova criada ainda.")
        return

    codigos = sorted(list(exams.keys()))
    selected = st.selectbox("Escolha a prova", codigos)

    exam = exams[selected]
    st.session_state.active_exam = exam

    existing = exam.get("gabarito")
    if isinstance(existing, list) and len(existing) == TOTAL_QUESTIONS:
        st.caption("Gabarito atual da prova carregado do banco.")
        if st.button("Usar gabarito atual como base"):
            st.session_state.draft_answer_key = [str(x).upper() for x in existing]

    for i in range(TOTAL_QUESTIONS):
        st.write(f"Q{i + 1}")
        cols = st.columns(len(CHOICES))
        for idx, choice in enumerate(CHOICES):
            pressed = cols[idx].button(choice, key=f"q{i+1}_{choice}")
            if pressed:
                st.session_state.draft_answer_key[i] = choice

        atual = st.session_state.draft_answer_key[i] or "(vazio)"
        st.caption(f"Marcada: {atual}")

    if st.button("Salvar gabarito"):
        if "" in st.session_state.draft_answer_key:
            st.warning("Preencha todas as questoes antes de salvar.")
            return

        exam["gabarito"] = st.session_state.draft_answer_key

        try:
            save_exam(exam)
        except OMRException as exc:
            st.error(str(exc))
            return

        st.success("Gabarito salvo com sucesso.")
        go_to_page("3. Corrigir gabaritos")


def grade_answers(gabarito: List[str], respostas: List[str]) -> Tuple[int, int]:
    acertos = 0
    valid_questions = min(len(gabarito), len(respostas))
    for i in range(valid_questions):
        if respostas[i] == gabarito[i]:
            acertos += 1
    return acertos, valid_questions


def build_result_dataframe(gabarito: List[str], scan: OMRScanResult) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(scan.questions):
        correct = gabarito[i] if i < len(gabarito) else "-"
        rows.append(
            {
                "questao": q.question,
                "gabarito": correct,
                "marcada": q.selected,
                "status_leitura": q.status,
                "confianca": round(q.confidence, 4),
                "acertou": q.selected == correct,
            }
        )
    return pd.DataFrame(rows)


def render_correction_page() -> None:
    st.subheader("3. Corrigir gabaritos")

    exam = st.session_state.active_exam
    if not exam:
        st.warning("Nenhuma prova ativa.")
        return

    gabarito = exam.get("gabarito", [])
    if not gabarito:
        st.warning("A prova selecionada ainda nao tem gabarito salvo.")
        return

    nome = st.text_input("Nome do aluno")
    turma = st.text_input("Turma")
    uploaded = st.file_uploader("Envie a foto da prova", type=["jpg", "jpeg", "png"])
    respostas_digitadas = st.text_input("Ou digite respostas manuais (ex: A B C D E ...)")
    show_debug = st.checkbox("Mostrar imagens de depuracao", value=True)

    if st.button("Corrigir"):
        respostas: List[str] = []
        scan: Optional[OMRScanResult] = None

        if uploaded is not None:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Arquivo de imagem invalido.")
                return

            try:
                scan = run_omr(image)
                respostas = scan.answers
            except OMRException as exc:
                st.error(str(exc))
                return
        else:
            respostas = [r.strip().upper() for r in respostas_digitadas.split() if r.strip()]

        if not respostas:
            st.warning("Nenhuma resposta encontrada.")
            return

        acertos, total = grade_answers(gabarito, respostas)
        percentual = round((acertos / total) * 100, 2) if total else 0.0

        st.success(f"Aluno: {nome or 'Nao informado'} | Turma: {turma or 'Nao informada'}")
        st.success(f"Resultado: {acertos}/{total} ({percentual}%)")
        st.write(f"Respostas lidas: {' '.join(respostas)}")
        st.write(f"Gabarito: {' '.join(gabarito)}")

        if scan is not None:
            df = build_result_dataframe(gabarito, scan)
            st.dataframe(df, use_container_width=True)
            em_duvida = int((df["status_leitura"] == "duvida").sum())
            em_branco = int((df["status_leitura"] == "em_branco").sum())
            if em_duvida > 0:
                st.warning(f"{em_duvida} questao(oes) com leitura em duvida. Revisao manual recomendada.")
            if em_branco > 0:
                st.info(f"{em_branco} questao(oes) detectadas em branco.")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar relatorio CSV",
                data=csv,
                file_name=f"resultado_{exam.get('codigo', 'prova')}_{nome or 'aluno'}.csv",
                mime="text/csv",
            )

            if show_debug:
                st.image(cv2.cvtColor(scan.page_view, cv2.COLOR_BGR2RGB), caption="Folha alinhada", use_container_width=True)
                st.image(cv2.cvtColor(scan.panel_view, cv2.COLOR_BGR2RGB), caption="Painel alinhado pelos marcadores", use_container_width=True)
                st.image(cv2.cvtColor(scan.binary_panel, cv2.COLOR_GRAY2RGB), caption="Painel binarizado", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Leitor de Gabarito", layout="wide")
    init_state()

    st.title("Leitor de Gabarito OMR")
    st.caption("Versao com leitura por imagem, alinhamento por marcadores e diagnostico de confianca.")

    pages = ["1. Criar prova", "2. Definir gabarito", "3. Corrigir gabaritos"]
    if st.session_state.nav_page not in pages:
        st.session_state.nav_page = pages[0]

    page = st.sidebar.radio("Etapas", pages, index=pages.index(st.session_state.nav_page))
    st.session_state.nav_page = page

    if page == pages[0]:
        render_create_exam_page()
    elif page == pages[1]:
        render_answer_key_page()
    else:
        render_correction_page()


if __name__ == "__main__":
    main()
