import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import random
import os
import uuid
import torch
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64
from groq import Groq
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from wordcloud import WordCloud, STOPWORDS
from huggingface_hub import hf_hub_download, upload_file
from streamlit_theme import st_theme

# --- CONFIGURAÇÃO HF ---
HF_TOKEN = st.secrets.get("HF_TOKEN")
REPO_ID = st.secrets.get("REPO_ID")
FILE_NAME = "feedback.csv"

# --- DETECÇÃO DE TEMA via streamlit-theme ---
_theme_data = st_theme()
_tema_base = (_theme_data.get("base", "light") if _theme_data else "light")
IS_DARK = (_tema_base == "dark")

TEMA = {
    "sidebar_bg":       "#1F1F21"   if IS_DARK else "#F5F7F6",
    "logo_path":        "LOGO PRESENTEIA DARK.png" if IS_DARK else "LOGO PRESENTEIA.png",
    "fig_bg":           "#1F1F21"   if IS_DARK else "#ffffff",
    "ax_bg":            "#2A2A2E"   if IS_DARK else "#ffffff",
    "spine_color":      "#444448"   if IS_DARK else "#333333",
    "tick_color":       "#888888"   if IS_DARK else "#aaaaaa",
    "label_color":      "#888888"   if IS_DARK else "#aaaaaa",
    "title_color":      "#CCCCCC"   if IS_DARK else "#555555",
    "grid_color":       "#3A3A3E"   if IS_DARK else "#cccccc",
    "scatter_base":     "#4A5568"   if IS_DARK else "#334466",
    "scatter_cmap":     "turbo",
    "star_color":       "#ff4b4b",
    "arrow_color":      "#ffcc00",
    "label_color_top5": "#ffcc00",
    "thead_bg":         "#2A2A2E"   if IS_DARK else "#f0f0f0",
    "thead_color":      "#CCCCCC"   if IS_DARK else "#000000",
    "border_color":     "#444448"   if IS_DARK else "#ddd",
    "cell_color":       "#CCCCCC"   if IS_DARK else "#000000",
    "intro_color":      "#BAC0C7"   if IS_DARK else "#404040"
}

# --- CORREÇÃO DE ENCODING ---
def corrigir_encoding(valor):
    if not isinstance(valor, str):
        return valor
    try:
        return valor.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return valor

def corrigir_encoding_df(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: corrigir_encoding(x) if isinstance(x, str) else x)
    return df

# --- FUNÇÕES DE MEMÓRIA OTIMIZADAS (FEEDBACK) ---
def obter_memoria_sessao():
    if "df_feedback_local" not in st.session_state:
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, repo_type="dataset", token=HF_TOKEN)
            df = pd.read_csv(path)
            if not df.empty:
                df['perfil_emb_vec'] = df['perfil_emb'].apply(
                    lambda x: np.fromstring(x.replace('[','').replace(']','').replace(',',' '), sep=' ')
                )
            st.session_state.df_feedback_local = df
        except Exception:
            st.session_state.df_feedback_local = pd.DataFrame(columns=["session_id", "p1_json", "p2_texto", "perfil_emb", "produto", "prod_emb", "voto"])
    return st.session_state.df_feedback_local

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_json_preview(data, limit=3):
    try:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            preview = dict(list(data.items())[:limit])
            if len(data) > limit:
                preview["..."] = f"Mais {len(data) - limit} chaves ocultas."
            return preview
        elif isinstance(data, list):
            preview = data[:limit]
            if len(data) > limit:
                preview.append(f"... (Mais {len(data) - limit} itens ocultos)")
            return preview
        return data
    except Exception as e:
        return {"erro": "Erro ao processar", "detalhes": str(e)}

def salvar_lote_feedback(lista_feedbacks):
    df_novos = pd.DataFrame(lista_feedbacks)
    if not df_novos.empty:
        df_novos['perfil_emb_vec'] = df_novos['perfil_emb'].apply(lambda x: np.array(x))
        st.session_state.df_feedback_local = pd.concat([st.session_state.df_feedback_local, df_novos], ignore_index=True)
    try:
        path_global = hf_hub_download(repo_id=REPO_ID, filename=FILE_NAME, repo_type="dataset", token=HF_TOKEN)
        df_global = pd.read_csv(path_global)
    except:
        df_global = pd.DataFrame(columns=["session_id", "p1_json", "p2_texto", "perfil_emb", "produto", "prod_emb", "voto"])
    df_sync = pd.concat([df_global, pd.DataFrame(lista_feedbacks)], ignore_index=True)
    df_sync.to_csv(FILE_NAME, index=False)
    upload_file(path_or_fileobj=FILE_NAME, path_in_repo=FILE_NAME, repo_id=REPO_ID, repo_type="dataset", token=HF_TOKEN)

def render_fb_table_completa(fb_completo, border_color, thead_bg, thead_color, cell_color):
    rows_html = ""
    for item in fb_completo:
        cor = "green" if "+" in item['alteracao_afinidade'] else "red"
        rows_html += (
            "<tr>"
            "<td style='padding:4px; border:1px solid " + border_color + ";'>" + str(item['produto']) + "</td>"
            "<td style='padding:4px; text-align:center; border:1px solid " + border_color + ";'>" + f"{item['afinidade_media_perfis']:.2f}" + "</td>"
            "<td style='padding:4px; text-align:center; border:1px solid " + border_color + "; color:" + cor + "; font-weight:bold;'>" + str(item['alteracao_afinidade']) + "</td>"
            "<td style='padding:4px; text-align:center; border:1px solid " + border_color + ";'>" + str(item['qtd_feedbacks']) + "</td>"
            "</tr>"
        )
    html = (
        "<div style='max-height: 300px; overflow-y: auto; border: 1px solid " + border_color + ";'>"
        "<table style='width:100%; border-collapse: collapse; font-size:0.85rem; color:" + cell_color + ";'>"
        "<thead style='position: sticky; top: 0; background:" + thead_bg + "; z-index: 1;'>"
        "<tr style='color:" + thead_color + ";'>"
        "<th style='padding:4px; text-align:left; border:1px solid " + border_color + ";'>Produto</th>"
        "<th style='padding:4px; text-align:center; border:1px solid " + border_color + ";'>Afinidade Média</th>"
        "<th style='padding:4px; text-align:center; border:1px solid " + border_color + ";'>Alteração</th>"
        "<th style='padding:4px; text-align:center; border:1px solid " + border_color + ";'>Nº Feedbacks</th>"
        "</tr>"
        "</thead>"
        "<tbody>" + rows_html + "</tbody>"
        "</table>"
        "</div>"
    )
    return html

# --- CONFIGURAÇÃO STREAMLIT ---
st.set_page_config(page_title="Recomendador de Presentes", layout="wide")

@st.cache_resource
def carregar_modelos():
    hf_token = st.secrets.get("HF_TOKEN")
    bi_id = "vg055/multilingual-e5-large-finetuned-IberAuTexTification2024-7030-task2-v2"
    cross_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
    bi_encoder = SentenceTransformer(bi_id, token=hf_token, trust_remote_code=True)
    try:
        cross_encoder = CrossEncoder(cross_id, token=hf_token)
    except:
        cross_encoder = None
    return bi_encoder, cross_encoder

@st.cache_data
def carregar_dados():
    if not os.path.exists("base_produtos_ouro.csv"):
        st.error("ERRO: Arquivo base_produtos_ouro.csv não encontrado")
        st.stop()

    df = None
    separadores = [';', ',', '\t', '|']
    for sep in separadores:
        for enc in ['latin-1', 'cp1252', 'utf-8-sig']:
            try:
                candidato = pd.read_csv(
                    "base_produtos_ouro.csv", sep=sep, engine='python',
                    on_bad_lines='skip', encoding=enc
                )
                if candidato is not None and len(candidato.columns) > 1:
                    df = candidato
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        st.error("ERRO CRÍTICO: Não foi possível ler base_produtos_ouro.csv.")
        st.stop()

    df.columns = [c.strip().replace('"', '').replace("'", "") for c in df.columns]
    df = corrigir_encoding_df(df)

    def limpar_preco_inteligente(valor):
        if pd.isna(valor) or str(valor).strip() == "": return 0.0
        v = str(valor).replace('"', '').strip()
        if ',' in v and '.' in v: return float(v.replace('.', '').replace(',', '.'))
        if ',' in v: return float(v.replace(',', '.'))
        if '.' in v:
            partes = v.split('.')
            if len(partes[-1]) == 3: return float(v.replace('.', ''))
            return float(v)
        try: return float(v)
        except: return 0.0

    def limpar_e_converter(x):
        clean_str = re.sub(r'[\[\]\n\r"\'\s]+', ' ', str(x)).strip()
        clean_str = clean_str.replace(',', ' ')
        try:
            lista_numbers = [float(n) for n in clean_str.split()]
            vec = np.array(lista_numbers, dtype=np.float32)
            if vec.size != 1024: return np.zeros(1024)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-10)
        except Exception:
            return np.zeros(1024)

    if 'embedding_perfil' in df.columns:
        df['embedding_perfil'] = df['embedding_perfil'].apply(limpar_e_converter)
    else:
        st.error("Erro Crítico: Coluna 'embedding_perfil' não encontrada.")
        st.stop()
    if 'preco' in df.columns:
        df['preco'] = df['preco'].apply(limpar_preco_inteligente)
    return df

model_bi, model_cross = carregar_modelos()
df_ouro = carregar_dados()

@st.cache_data
def calcular_pca_produtos(_df_ouro):
    embs_produtos = np.vstack(_df_ouro['embedding_perfil'].values)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs_produtos)
    return coords, pca

pca_produtos_coords, pca_global = calcular_pca_produtos(df_ouro)

if "session_id" not in st.session_state: st.session_state.session_id = str(uuid.uuid4())
if "historico_inputs" not in st.session_state: st.session_state.historico_inputs = []
if "votos_temp" not in st.session_state: st.session_state.votos_temp = {}
if "banidos_sessao" not in st.session_state: st.session_state.banidos_sessao = set()
if "liked_sessao" not in st.session_state: st.session_state.liked_sessao = set()
if "dados_ultima_busca" not in st.session_state: st.session_state.dados_ultima_busca = None
if "ranking_atual" not in st.session_state: st.session_state.ranking_atual = None
if "feedback_enviado" not in st.session_state: st.session_state.feedback_enviado = False
if "houve_mudanca" not in st.session_state: st.session_state.houve_mudanca = True
if "tela_atual" not in st.session_state: st.session_state.tela_atual = "recomendacao"
if "num_inputs" not in st.session_state: st.session_state.num_inputs = 0

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(f"""
        <style>
            [data-testid="stSidebar"] {{
                background-color: {TEMA["sidebar_bg"]};
            }}
            [data-testid="stSidebar"] img {{
                width: 100% !important;
                border-radius: 12px;
            }}
        </style>
    """, unsafe_allow_html=True)

    logo_path = TEMA["logo_path"]
    if os.path.exists(logo_path):
        st.image(logo_path)
    elif os.path.exists("LOGO PRESENTEIA.png"):
        st.image("LOGO PRESENTEIA.png")

    st.header("⚙️ Opções")

    if st.session_state.tela_atual == "recomendacao":
        if st.button("🔄 TENTAR OUTRA PESSOA", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
        if st.button("📋 LISTA DE ITENS", use_container_width=True, type="secondary"):
            st.session_state.tela_atual = "lista_itens"
            st.rerun()
        if st.button("👤 SOBRE O CRIADOR", use_container_width=True, type="secondary"):
            st.session_state.tela_atual = "sobre_criador"
            st.rerun()
    else:
        if st.button("🎁 BUSCAR RECOMENDAÇÃO", use_container_width=True, type="secondary"):
            st.session_state.tela_atual = "recomendacao"
            st.rerun()
        if st.button("📋 LISTA DE ITENS", use_container_width=True, type="secondary"):
            st.session_state.tela_atual = "lista_itens"
            st.rerun()
        if st.button("👤 SOBRE O CRIADOR", use_container_width=True, type="secondary"):
            st.session_state.tela_atual = "sobre_criador"
            st.rerun()

    st.divider()
    st.subheader("Histórico da conversa")
    if st.session_state.historico_inputs:
        for idx, texto in enumerate(st.session_state.historico_inputs):
            st.info(f"**{idx+1}.** {texto}")
    else: st.write("Nenhuma descrição enviada.")

# --- TELA DE LISTA DE ITENS ---
if st.session_state.tela_atual == "lista_itens":
    st.title("📋 Lista de Itens")

    df_feedback = obter_memoria_sessao()

    stats_feedback = df_feedback.groupby('produto').agg(
        total_feedbacks=('voto', 'count'),
        upvotes=('voto', lambda x: (x == 'Sim').sum())
    ).reset_index()

    df_display = df_ouro[['produto', 'preco', 'marketplace','link']].copy()
    df_display = df_display.merge(stats_feedback, on='produto', how='left')
    df_display['total_feedbacks'] = df_display['total_feedbacks'].fillna(0).astype(int)
    df_display['upvotes'] = df_display['upvotes'].fillna(0).astype(int)
    df_display = df_display.sort_values('upvotes', ascending=False)
    df_display = df_display.rename(columns={
        'produto': 'Produto',
        'preco': 'Valor (R$)',
        'link': 'Link',
        'total_feedbacks': 'Feedbacks',
        'upvotes': 'Upvotes'
    })

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Link"),
            "Valor (R$)": st.column_config.NumberColumn("Valor (R$)", format="R$ %.2f")
        }
    )

    st.stop()

# --- TELA SOBRE O CRIADOR ---
if st.session_state.tela_atual == "sobre_criador":
    st.title("👤 Sobre o Criador")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            bin_str = get_base64_of_bin_file('foto_hugo.jpg')
            st.markdown(
                f"""
                <div style="display: flex; justify-content: center;">
                    <img src="data:image/jpeg;base64,{bin_str}" 
                         style="border-radius: 50%; width: 250px; height: 250px; object-fit: cover; border: 4px solid {TEMA['intro_color']};">
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.image("foto_hugo.jpg", width=300)

    with col2:
        st.markdown(f"""
        <div style="color: {TEMA['intro_color']}; margin-top: 10px;">
            <h2>Hugo Rocha</h2>
            <p>Engenheiro Mecatrônico formado pelo Insper com especialização em Data Science e Analytics pela ESALQ-USP, com carreira desenvolvida em áreas estratégicas de Supply Chain, Logística e Inteligência de Dados. Entusiasta de Data Science e sempre inventando uns projetos 🤪.</p>
            <p>📧 <b>Email: </b>hugoncalves@outlook.com</p>
            <p>🔗 <a href="https://www.linkedin.com/in/hugogrocha" target="_blank" style="color:#00bcd4;">Linkedin</a></p>
            <p>🐙 <a href="https://github.com/hugogr99" target="_blank" style="color:#00bcd4;">Github</a></p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div style="color: {TEMA['intro_color']};">
        <h3>🎁 Sobre o presenteIA</h3>
        <p>A ideia para este projeto surgiu com as aulas de Natural Language Processing no MBA. Foi na época de natal e eu pensei em unir o útil ao agradável, criando um motor de recomendação de presentes que entendesse características do presenteado e fizesse sugestões a partir disso.</p>
        <p>O funcionamento é muito simples: primeiro, eu criei uma base de dados de produtos de diferentes players de ecommerce. Por meio da API do <a href="https://console.groq.com/home" target="_blank" style="color:#00bcd4;">Groq</a>, utilizei modelos de IA Generativa para criar perfis de pessoas que gostariam de receber cada item de presente. Em seguida, uso a biblioteca Sentence-Transformers para transformar o perfil em vetores (embeddings). Quando o usuário insere informações, a IA gera um perfil, transforma esse perfil em vetores e utiliza similaridade de cossenos para dar "match" entre produto e presenteado!</p>
        <p>A melhor parte é que o usuário também pode ajudar a melhorar o algoritmo! Implementei um sistema simples de Q-learning onde o usuário avalia as melhores e piores opções e essas avaliações são consideradas para novos perfis parecidos.</p>
        <p>Me diverti muito com este projeto, espero que ache divertido também! :)
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# --- INTERFACE PRINCIPAL (TELA DE RECOMENDAÇÃO) ---
header_placeholder = st.empty()
progress_placeholder = st.empty()

if "input_atual" in st.session_state and st.session_state.ranking_atual is None:
    if "progresso_atual" not in st.session_state:
        st.session_state.progresso_atual = 0
    if "dots_count" not in st.session_state:
        st.session_state.dots_count = 1

    dots = "." * st.session_state.dots_count
    header_placeholder.title(f"🤔 PENSANDO{dots} ({st.session_state.progresso_atual}%)")
else:
    header_placeholder.title("🎁 presenteIA - Recomendador de Presentes")
    if "input_atual" not in st.session_state:
        st.markdown(f"""
        <div style="color: {TEMA['intro_color']}; margin-top: 20px;">
            <h3>E aí, beleza? Eu sou o presenteIA e vou te ajudar a escolher um presente para aquela pessoa especial!</h3>
            <p>Olha só como eu funciono:</p>
            <ol>
                <li><b>Descreva a pessoa</b> que vai receber o presente no campo abaixo — me fale sobre hobbies, características, gostos, idade, ocasião e quanto você quer gastar.</li>
                <li><b>Aguarde a análise</b> — vou montar o perfil da pessoa e buscar os presentes mais adequados.</li>
                <li><b>Veja as recomendações e avalie</b> - você pode clicar em ❤️ ou ❌ para avaliar cada sugestão.</li>
                <li><b>Envie o feedback</b> - depois de avaliar, clique em "Enviar Feedback" para adicionar suas opiniões ao meu banco de dados. Com isso, você vai me ajudar a fazer recomendações melhores :)</li>
                <li><b>Recalcular</b> - não gostou do que sugeri? Tudo bem, errar é robótico! Clique em "Recalcular Sugestões" depois de ter dado feedbacks para refinar os resultados.</li>
                <li><b>Novo perfil</b> - acesse o menu lateral e clique em "Tentar outra pessoa" para zerar os inputs e recomeçar.</li>
                <li><b>Lista de itens</b> - acesse todos os itens do banco de dados pelo menu lateral, clicando em "Lista de Itens".</li>
            </ol>
            <p>💡 <b>Lembre-se: quanto mais detalhes você der, melhor será a recomendação!</b><br>
            Ah, lembre-se também que eu sou um algoritmo, então posso errar às vezes!</p>
            <h3>Boas recomendações!</h3>
        </div>
        """, unsafe_allow_html=True)

novo_dado = st.chat_input("Descreva a pessoa... (hobbies, características, gostos)")
if novo_dado:
    st.session_state.historico_inputs.append(novo_dado)
    st.session_state.votos_temp = {}
    partes = [f"input {i+1}: {txt}" for i, txt in enumerate(st.session_state.historico_inputs)]
    st.session_state.input_atual = ", ".join(partes)
    st.session_state.ranking_atual = None
    st.session_state.feedback_enviado = False
    st.session_state.houve_mudanca = True
    st.session_state.progresso_atual = 0
    st.session_state.dots_count = 1
    st.session_state.num_inputs += 1
    st.rerun()

if "input_atual" in st.session_state:
    if st.session_state.ranking_atual is None:
        input_final = st.session_state.input_atual
        chaves_validas = [k.strip() for k in [st.secrets.get("GROQ_KEY_1"), st.secrets.get("GROQ_KEY_2"), st.secrets.get("GROQ_KEY_3"), st.secrets.get("GROQ_KEY_4")] if k]

        st.markdown("""
        <style>
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .fade-in { animation: fadeIn 0.8s ease-in; }
        </style>
        """, unsafe_allow_html=True)

        # Durante o PENSANDO: gráfico ocupa 2 colunas, perfil 1, feedbacks 1.
        # O debug (P3/P4) NÃO aparece aqui — fica só no expander pós-resultado.
        col_grafico, col_perfil, col_feedbacks = st.columns([2, 1.5, 1.5])

        with col_grafico:
            graph_placeholder = st.empty()

        with col_perfil:
            st.markdown("#### MONTANDO O PERFIL...")
            p1_placeholder = st.empty()  # JSON do perfil (P1)
            p2_placeholder = st.empty()  # Texto descritivo (P2)

        with col_feedbacks:
            fb_titulo_placeholder = st.empty()
            fb_placeholder = st.empty()
            # MUDANÇA 2: debug_titulo e debug_placeholder declarados mas nunca
            # preenchidos durante o PENSANDO — só aparecem no expander após rerun.
            debug_titulo_placeholder = st.empty()
            debug_placeholder = st.empty()

        def render_grafico(
            coords_todos,
            star_pos=None,
            scores_norm=None,
            top5_indices=None,
            nomes_top5=None,
            titulo="Mapeando produtos...",
        ):
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.set_facecolor(TEMA["ax_bg"])
            fig.patch.set_facecolor(TEMA["fig_bg"])
            ax.tick_params(colors=TEMA["tick_color"], labelsize=7)
            ax.xaxis.label.set_color(TEMA["label_color"])
            ax.yaxis.label.set_color(TEMA["label_color"])
            for spine in ax.spines.values():
                spine.set_edgecolor(TEMA["spine_color"])
            ax.grid(True, linestyle='--', alpha=0.2, color=TEMA["grid_color"])
            ax.set_xlabel("Semântica Geral", fontsize=8, color=TEMA["label_color"])
            ax.set_ylabel("Relevância de Perfil", fontsize=8, color=TEMA["label_color"])
            ax.set_title(titulo, fontsize=9, color=TEMA["title_color"], pad=8)

            if scores_norm is not None:
                scores_amplificados = np.power(np.clip(scores_norm, 0, 1), 0.35)
                ax.scatter(
                    coords_todos[:, 0], coords_todos[:, 1],
                    c=scores_amplificados, cmap=TEMA["scatter_cmap"], s=22, alpha=0.85,
                    edgecolors='none', vmin=0, vmax=1
                )
            else:
                ax.scatter(
                    coords_todos[:, 0], coords_todos[:, 1],
                    c=TEMA["scatter_base"], s=14, alpha=0.45, edgecolors='none'
                )

            if star_pos is not None:
                ax.scatter(
                    star_pos[0], star_pos[1],
                    marker='*', s=520, color=TEMA["star_color"],
                    edgecolors='white', linewidths=0.8, zorder=10
                )

            if top5_indices is not None and star_pos is not None and nomes_top5 is not None:
                for idx_t5, nome in zip(top5_indices, nomes_top5):
                    px, py = coords_todos[idx_t5]
                    ax.annotate(
                        '',
                        xy=(px, py), xytext=(star_pos[0], star_pos[1]),
                        arrowprops=dict(
                            arrowstyle='->', color=TEMA["arrow_color"],
                            lw=1.6, alpha=0.85,
                            connectionstyle='arc3,rad=0.08'
                        ),
                        zorder=8
                    )
                    ax.text(
                        px + 0.04, py + 0.04, nome[:10],
                        fontsize=7.5, color=TEMA["label_color_top5"],
                        fontweight='bold', zorder=9
                    )

            plt.tight_layout()
            return fig

        def ease(x):
            return x * x * (3 - 2 * x)

        def atualizar_progresso(valor):
            st.session_state.progresso_atual = int(valor)
            st.session_state.dots_count = (st.session_state.dots_count % 3) + 1
            dots = "." * st.session_state.dots_count
            header_placeholder.title(f"🤔 PENSANDO{dots:3s} ({st.session_state.progresso_atual}%)")

        atualizar_progresso(5)
        fig_inicial = render_grafico(coords_todos=pca_produtos_coords, titulo="Mapeando base de produtos...")
        graph_placeholder.pyplot(fig_inicial)
        plt.close(fig_inicial)

        sucesso_api = False
        res_tabela_raw, res_final, res_j_raw = "", "", ""
        dados_json = {}
        client = None

        atualizar_progresso(10)
        while len(chaves_validas) > 0 and not sucesso_api:
            chave_atual = random.choice(chaves_validas)
            try:
                client = Groq(api_key=chave_atual)
                p1_placeholder.write("Processando Prompt 1...")
                atualizar_progresso(15)
                
                # --- PROMPT 1: cria JSONs a partir dos inputs ---
                prompt_tabela = '''
                Estou criando um recomendador de presentes e você vai me ajudar.
                    Estamos na fase de identificação do perfil da pessoa que será presenteada.
                    Sua missão é extrair dados para um JSON de perfil.
                    REGRAS OBRIGATÓRIAS:
                    1. Retorne APENAS o objeto JSON.
                    2. Use apenas estas chaves (todas referentes à pessoa presenteada): "nome", "ocasiao", "hobbies", "caracteristicas", "valor", "sexo", "idade", "incompativel", "necessidade".
                    3. O campo 'necessidade' só deve ser preenchido se for EXPLICITAMENTE dado pelo usuário com o uso de palavras como "precisa", "necessita", "quer" e deve ser literal. Não pode haver nenhuma associação. Caso contrário, mantenha vazio!
                    4. É PROIBIDO incluir chaves sem dados. Caso um campo não tenha dados, ele não deve aparecer no JSON final.
                    5. É PROIBIDO fazer qualquer tipo de associação ou suposição. Você deve ser absolutamente literal com as informações que foram dadas.
                    6. CASO ESPECIFICO 1: o usuário disser que presenteado precisa EXPLICITAMENTE de um celular, substitua por "smartphone" na necessidade. Caso seja um acessório de celular (capinha, tripé, etc), mantenha como celular!
                    7. CASO ESPECIFICO 2: o usuário disser que presenteado precisa EXPLICITAMENTE de uma televisão, tv ou televisor, substitua por "smart tv" na necessidade.
                '''
                res_tabela_raw = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "system", "content": prompt_tabela}, {"role": "user", "content": input_final}],
                    response_format={"type": "json_object"}
                ).choices[0].message.content
                dados_json = json.loads(res_tabela_raw)
                p1_placeholder.json(dados_json)

                # --- PROMPT 2: cria breve parágrafo sobre perfil a partir do JSON do Prompt 1 ---
                atualizar_progresso(25)
                p2_placeholder.write("Processando Prompt 2...")
                prompt_texto = '''
                Atue como um analista de profiles. Transcreva as características presentes no JSON em um parágrafo descritivo de no máximo 300 caracteres.
                Você deve escrever apenas o que está no JSON, sem inventar nem supor mais nada.
                A resposta deve seguir o exemplo: "A pessoa se chama {nome}. Tem {idade} anos. Vai comemorar {ocasiao}. Tem como hobbies: {hobbies}. Tem como características: {características}. Precisa de {necessidade}. Não gosta de {incompativel}. O valor é de até {valor}. {Descreva 3 adjetivos possíveis da pessoa com base nas CARACTERÍSTICAS e HOBBIES APENAS (se não tiver essas chaves, não retorne nada). SEM INTRODUÇÃO, seguindo o modelo: adj1, adj2, adj3}"
                Dados que estejam vazios, com null, 'não informado' não podem ser mencionados na resposta e devem ter sua parte na estrutura ocultada.
                '''
                res_final = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": prompt_texto}, {"role": "user", "content": res_tabela_raw}]
                ).choices[0].message.content
                p2_placeholder.info(res_final)

                atualizar_progresso(35)
                ultimo_input = st.session_state.historico_inputs[-1] if st.session_state.historico_inputs else ""
                necessidade = str(dados_json.get('necessidade', ''))
                hobbies = str(dados_json.get('hobbies', ''))
                v_necessidade = model_bi.encode([f"query: {necessidade}. {hobbies}"], normalize_embeddings=True)[0]
                v_ultimo = model_bi.encode([f"query: {ultimo_input}"], normalize_embeddings=True)[0]
                v_busca = (v_necessidade * 0.7) + (v_ultimo * 0.3)
                v_busca = v_busca / (np.linalg.norm(v_busca) + 1e-10)
                star_pos_pca = pca_global.transform(v_busca.reshape(1, -1))[0]

                atualizar_progresso(40)
                fig_com_estrela = render_grafico(
                    coords_todos=pca_produtos_coords,
                    star_pos=star_pos_pca,
                    titulo="Perfil identificado! Calculando afinidades..."
                )
                graph_placeholder.pyplot(fig_com_estrela)
                plt.close(fig_com_estrela)

                sucesso_api = True
            except:
                chaves_validas.remove(chave_atual)

        ultimo_input = st.session_state.historico_inputs[-1] if st.session_state.historico_inputs else ""
        necessidade = str(dados_json.get('necessidade', ''))
        hobbies = str(dados_json.get('hobbies', ''))
        tokens_necessidade = [t.lower() for t in re.findall(r'\w{4,}', necessidade)]

        if not sucesso_api:
            v_necessidade = model_bi.encode([f"query: {necessidade}. {hobbies}"], normalize_embeddings=True)[0]
            v_ultimo = model_bi.encode([f"query: {ultimo_input}"], normalize_embeddings=True)[0]
            v_busca = (v_necessidade * 0.7) + (v_ultimo * 0.3)
            v_busca = v_busca / (np.linalg.norm(v_busca) + 1e-10)
            star_pos_pca = pca_global.transform(v_busca.reshape(1, -1))[0]

        genero_usuario = str(dados_json.get("sexo", "")).lower()
        idade_val = dados_json.get("idade")
        categoria_usuario = "adulto"
        if idade_val:
            try:
                idade_int = int(re.search(r'\d+', str(idade_val)).group())
                if idade_int <= 10: categoria_usuario = "criança"
                elif 10 < idade_int < 18: categoria_usuario = "adolescente"
                else: categoria_usuario = "adulto"
            except: pass

        budget = float(re.sub(r'[^\d.]', '', str(dados_json.get('valor', '999999')).replace(',', '.')) or 999999.0)
        if budget > 0:
            df_base_busca = df_ouro[df_ouro['preco'] <= budget].copy()
        else:
            df_base_busca = df_ouro.copy()

        indices_filtrados = df_base_busca.index.tolist()
        pca_coords_filtradas = pca_produtos_coords[indices_filtrados]

        df_mem_calculo = obter_memoria_sessao()
        termos_odiados = set(re.findall(r'\w{3,}', str(dados_json.get("incompativel", "")).lower()))
        ajustes_produtos_map = {}

        fb_titulo_placeholder.markdown("#### COLHENDO FEEDBACKS ANTIGOS")
        atualizar_progresso(45)

        _hobbies_str      = str(dados_json.get('hobbies', '')).strip()
        _caract_str       = str(dados_json.get('caracteristicas', '')).strip()
        _necessidade_str  = str(dados_json.get('necessidade', '')).strip()
        _ocasiao_str      = str(dados_json.get('ocasiao', '')).strip()
        perfil_tem_substancia = any([
            _hobbies_str and _hobbies_str.lower() not in ('none', 'null', ''),
            _caract_str  and _caract_str.lower()  not in ('none', 'null', ''),
            _necessidade_str and _necessidade_str.lower() not in ('none', 'null', ''),
            _ocasiao_str and _ocasiao_str.lower() not in ('none', 'null', ''),
        ])

        def calcular_score_inicial(row):
            prod_nome = str(row['produto']).lower()
            if prod_nome in st.session_state.banidos_sessao:
                return -1.0
            ctx = f"{prod_nome} {str(row.get('perfil_presenteado', '')).lower()}"
            for p in termos_odiados:
                if p in ctx:
                    return -1.0
            score = np.dot(v_busca, row['embedding_perfil'])
            if any(x in necessidade.lower() or x in ultimo_input.lower() for x in ["celular", "smartphone"]):
                if "celular" in prod_nome or "smartphone" in prod_nome:
                    score += 1.5
            for token in tokens_necessidade:
                if token in prod_nome:
                    score += 0.8
            if categoria_usuario == "criança":
                if not any(x in ctx for x in ["criança", "infantil", "kids", "brinquedo", "baby", "anos"]):
                    return -1.0
            elif categoria_usuario == "adolescente":
                if any(x in ctx for x in ["bebê", "baby", "brinquedo para bebê", "maral"]):
                    return -1.0
            elif categoria_usuario == "adulto":
                if any(x in ctx for x in ["criança", "infantil", "bebê", "baby", "brinquedo para bebê", "maral"]):
                    if "adulto" not in ctx and "unissex" not in ctx:
                        return -1.0

            if perfil_tem_substancia:
                if "masc" in genero_usuario or "homem" in genero_usuario:
                    if any(x in ctx for x in ["feminino", "feminina", "maquiagem", "batom", "esmalte", "creme facial feminino"]):
                        if "unissex" not in ctx:
                            return -1.0
                elif "fem" in genero_usuario or "mulher" in genero_usuario:
                    if any(x in ctx for x in ["masculino", "masculina", "barba", "aparador de barba", "creme pós-barba"]):
                        if "unissex" not in ctx:
                            return -1.0

            if row['produto'] in st.session_state.liked_sessao:
                score += 3.0
            ajuste_fb = ajustes_produtos_map.get(row['produto'], 0.0)
            return score + ajuste_fb

        df_f = df_base_busca.copy()
        df_f['score_bi'] = df_f.apply(calcular_score_inicial, axis=1)

        scores_bi_all = np.zeros(len(df_ouro))
        for orig_idx, score_val in zip(df_f.index, df_f['score_bi'].values):
            scores_bi_all[orig_idx] = np.clip(score_val, 0, None)
        s_max = scores_bi_all.max()
        scores_bi_norm = scores_bi_all / (s_max + 1e-9)

        FRAMES_FASE_A = 6
        SLEEP_FASE_A = 0.08

        atualizar_progresso(50)
        for f in range(FRAMES_FASE_A + 1):
            t = f / FRAMES_FASE_A
            alpha = ease(t)
            fig_frame = render_grafico(
                coords_todos=pca_produtos_coords,
                star_pos=star_pos_pca,
                scores_norm=scores_bi_norm * alpha,
                titulo="Calculando afinidade semântica...",
            )
            graph_placeholder.pyplot(fig_frame)
            plt.close(fig_frame)
            time.sleep(SLEEP_FASE_A)

        fb_html = ""
        tabela_feedback_dados = []
        tabela_feedback_completa = []

        atualizar_progresso(55)
        fb_placeholder.write("Analisando feedbacks anteriores...")
        if not df_mem_calculo.empty and 'voto' in df_mem_calculo.columns:
            matriz_feedbacks = np.stack(df_mem_calculo['perfil_emb_vec'].values)
            similaridades = np.dot(matriz_feedbacks, v_busca)
            votos_pesos = np.where(df_mem_calculo['voto'] == "Sim", 0.1, -0.3)
            df_mem_calculo['efeito'] = similaridades * votos_pesos
            df_mem_calculo['afinidade_perfil'] = similaridades

            ajustes_produtos_map = df_mem_calculo.groupby('produto')['efeito'].sum().to_dict()

            agrupado_completo = df_mem_calculo.groupby('produto', as_index=False).agg(
                influencia_total=('efeito', 'sum'),
                afinidade_media_perfis=('afinidade_perfil', 'mean'),
                qtd_feedbacks=('efeito', 'count'),
            )
            agrupado_completo = agrupado_completo.sort_values('influencia_total', key=lambda x: x.abs(), ascending=False)
            agrupado_completo['alteracao_afinidade'] = agrupado_completo['influencia_total'].apply(
                lambda x: f"+{x*100:.1f}%" if x > 0 else f"{x*100:.1f}%"
            )
            tabela_feedback_completa = agrupado_completo[['produto', 'afinidade_media_perfis', 'alteracao_afinidade', 'qtd_feedbacks']].to_dict('records')

            negativos = agrupado_completo[agrupado_completo['influencia_total'] < 0]
            positivos = agrupado_completo[agrupado_completo['influencia_total'] > 0]

            # ALTERAÇÃO 1: Apenas 2 negativos e 2 positivos (em vez de 3 e 3)
            sample_neg = negativos.sample(n=min(2, len(negativos))) if len(negativos) > 0 else pd.DataFrame()
            sample_pos = positivos.sample(n=min(2, len(positivos))) if len(positivos) > 0 else pd.DataFrame()

            sample_4 = pd.concat([sample_neg, sample_pos], ignore_index=True)
            sample_4 = sample_4.sample(frac=1).reset_index(drop=True)

            tabela_feedback_dados = sample_4[['produto', 'afinidade_media_perfis', 'alteracao_afinidade', 'qtd_feedbacks']].to_dict('records')

            border = TEMA['border_color']
            thead_bg = TEMA['thead_bg']
            thead_color = TEMA['thead_color']
            cell_color = TEMA['cell_color']

            rows = ""
            for item in tabela_feedback_dados:
                cor = "green" if "+" in item['alteracao_afinidade'] else "red"
                rows += (
                    "<tr>"
                    "<td style='padding:4px; border:1px solid " + border + ";'>" + str(item['produto']) + "</td>"
                    "<td style='padding:4px; text-align:center; border:1px solid " + border + ";'>" + f"{item['afinidade_media_perfis']:.2f}" + "</td>"
                    "<td style='padding:4px; text-align:center; border:1px solid " + border + "; color:" + cor + "; font-weight:bold;'>" + str(item['alteracao_afinidade']) + "</td>"
                    "<td style='padding:4px; text-align:center; border:1px solid " + border + ";'>" + str(item['qtd_feedbacks']) + "</td>"
                    "</tr>"
                )

            fb_html = (
                "<table style='width:100%; border-collapse: collapse; font-size:0.85rem; color:" + cell_color + ";'>"
                "<thead><tr style='background:" + thead_bg + "; color:" + thead_color + ";'>"
                "<th style='padding:4px; text-align:left; border:1px solid " + border + ";'>Produto</th>"
                "<th style='padding:4px; text-align:center; border:1px solid " + border + ";'>Afinidade Média</th>"
                "<th style='padding:4px; text-align:center; border:1px solid " + border + ";'>Alteração</th>"
                "<th style='padding:4px; text-align:center; border:1px solid " + border + ";'>Nº Feedbacks</th>"
                "</tr></thead><tbody>" + rows + "</tbody></table>"
                "<p style='margin-top:8px;'><small>" + str(len(df_mem_calculo)) + " feedbacks analisados no total.</small></p>"
            )
            fb_placeholder.markdown(fb_html, unsafe_allow_html=True)
        else:
            fb_placeholder.write("Nenhum feedback prévio encontrado.")

        atualizar_progresso(60)
        df_f['score_bi'] = df_f.apply(calcular_score_inicial, axis=1)
        top_rec = df_f[df_f['score_bi'] > 0.01].sort_values("score_bi", ascending=False).head(150).copy()

        atualizar_progresso(65)
        if not top_rec.empty and model_cross is not None:
            query_limpa = f"query: {necessidade}. {ultimo_input}"
            pares = [[query_limpa, f"{r['produto']} {r['perfil_presenteado']}"] for _, r in top_rec.iterrows()]
            top_rec['score_cross'] = model_cross.predict(pares)
            c_min, c_max = top_rec['score_cross'].min(), top_rec['score_cross'].max()
            top_rec['score_cross_norm'] = (top_rec['score_cross'] - c_min) / (c_max - c_min + 1e-6)
        else:
            top_rec['score_cross_norm'] = 0.1

        atualizar_progresso(70)
        
        # --- PROMPT 3: valida o resultado dos encoders e atribui uma nota ---
        if not top_rec.empty:
            top_llm = top_rec.sort_values("score_cross_norm", ascending=False).head(25).reset_index(drop=True)
            lista_detalhes = [f"ID {i}: {r['produto']} - Perfil: {str(r['perfil_presenteado'])[:250]}..." for i, r in top_llm.iterrows()]

            dados_json_sem_sexo = {k: v for k, v in dados_json.items() if k != "sexo"}

            prompt_julgamento = f'''
                Você é a pessoa descrita no perfil a seguir e será presenteada um dos itens da lista.
                Perfil: {json.dumps(dados_json_sem_sexo, ensure_ascii=False)}. Necessidade: {necessidade}. Orçamento: R${budget}.
                Analise os 25 produtos e retorne um JSON com TODOS os 25 produtos avaliados. 
                Caso perfil não tenha hobbies, características, necessidades e ocasião, todos os produtos devem receber 0. As notas devem crescer levemente quanto mais completo for o perfil.
                A ordem de prioridade para dar notas, da maior para menor é: NECESSIDADE > HOBBIES > CARACTERÍSTICAS > OCASIÃO. Gênero/sexo NÃO é critério de nota — jamais use gênero para favorecer ou penalizar produtos. Só considere gênero se o próprio hobbie ou necessidade indicar isso claramente (ex: hobbies de maquiagem indicam interesse em cosméticos, independente do gênero).
                Forneça uma frase entre 100 e 200 caracteres como se você fosse a pessoa do perfil reagindo ao presente para TODOS os produtos que tiverem a nota maior ou igual a 5.0. Estrutura: "Interjeição (pode ser Uau, Nossa, etc) + (falar o que é o produto)+"!"+ Eu achei bem" +""(opinião do perfil sobre o produto e sobre a utilidade dele na vida da pessoa)"!
                Para os demais (nota menor ou igual a 4.9), deixe a justificativa vazia.
                REGRA DE OURO: Itens muito banais ou que geralmente não sejam presentes ou que sejam muito baratos devem receber uma nota baixa, a não ser que explicitamente pedidos nas necessidades.
                FORMATO OBRIGATÓRIO: {{"resultados": [{{"id": 0, "nota": 10.0, "justificativa": "..."}}, {{"id": 1, "nota": 9.5, "justificativa": ""}}, ...]}}
                IMPORTANTE: Retorne EXATAMENTE 25 itens (id de 0 a 24). Evite repetições.
            '''

            atualizar_progresso(75)

            chaves_validas_p3 = [k.strip() for k in [st.secrets.get("GROQ_KEY_1"), st.secrets.get("GROQ_KEY_2"), st.secrets.get("GROQ_KEY_3"), st.secrets.get("GROQ_KEY_4")] if k]

            if st.session_state.num_inputs == 1:
                modelos_disponiveis = [
                    "llama-3.3-70b-versatile",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-3.1-8b-instant"
                ]
            else:
                modelos_disponiveis = [
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-3.1-8b-instant"
                ]

            modelo_usado = None
            res_j_raw = None
            sucesso_llm = False

            for modelo_atual in modelos_disponiveis:
                if sucesso_llm:
                    break

                chaves_tentativa = chaves_validas_p3.copy()
                while len(chaves_tentativa) > 0:
                    chave_atual = random.choice(chaves_tentativa)
                    try:
                        client = Groq(api_key=chave_atual)

                        chat_completion = client.chat.completions.create(
                            model=modelo_atual,
                            messages=[
                                {"role": "system", "content": "Você é um consultor que responde estritamente em JSON."},
                                {"role": "user", "content": prompt_julgamento + "\n" + "\n".join(lista_detalhes)}
                            ],
                            response_format={"type": "json_object"}
                        )

                        res_j_raw = chat_completion.choices[0].message.content
                        finish_reason = chat_completion.choices[0].finish_reason

                        if finish_reason != "stop":
                            raise Exception(f"API não terminou: {finish_reason}")

                        dados_fase3 = json.loads(res_j_raw)
                        resultados = dados_fase3.get("resultados", [])

                        ids_recebidos = set([int(item['id']) for item in resultados if 'id' in item])
                        ids_esperados = set(range(25))

                        if ids_recebidos != ids_esperados:
                            raise Exception(f"Modelo retornou apenas {len(resultados)} produtos em vez de 25")

                        modelo_usado = modelo_atual
                        sucesso_llm = True
                        break

                    except Exception as e:
                        chaves_tentativa.remove(chave_atual)
                        continue

            if not sucesso_llm or modelo_usado is None:
                st.error("ERRO CRÍTICO: Todos os modelos e chaves falharam no Prompt 3")
                st.stop()

            dados_fase3 = json.loads(res_j_raw)
            resultados = dados_fase3.get("resultados", [])

            notas_map = {int(item['id']): item['nota'] for item in resultados}
            just_map  = {int(item['id']): item.get('justificativa', "") for item in resultados}
            top_llm['score_llm']     = top_llm.index.map(notas_map).astype(float) / 10
            top_llm['justificativa'] = top_llm.index.map(just_map)

            # ALTERAÇÃO 2: Mostrar apenas os 2 primeiros produtos do debug_json
            debug_json_preview = []
            top_llm_sorted_temp = top_llm.sort_values("score_llm", ascending=False)
            for idx, row in top_llm_sorted_temp.iterrows():
                if len(debug_json_preview) >= 2: 
                    break
                debug_json_preview.append({
                    "produto": row['produto'],
                    "score_encoder": float(row['score_cross_norm']),
                    "score_llm": float(row['score_llm']),
                    "justificativa": row['justificativa'] if row['justificativa'] else ""
                })
            
            # Mostrar o preview no "PENSANDO"
            debug_titulo_placeholder.markdown("#### REFINANDO RESULTADOS...")
            debug_placeholder.json(debug_json_preview)
            
            # Sleep de 1 segundo após mostrar o preview
            time.sleep(1)

            atualizar_progresso(85)
            
            # --- PROMPT 4: Ajusta falhas do Prompt 3 ---
            
            NOTA_MINIMA_JUST = 5.0
            produtos_sem_just = top_llm[
                (top_llm['score_llm'] >= NOTA_MINIMA_JUST / 10) &
                (top_llm['justificativa'].apply(lambda x: not str(x).strip()))
            ].copy()

            if not produtos_sem_just.empty:
                lista_p4 = [
                    f"ID {row_idx}: {row['produto']} - Perfil do produto: {str(row['perfil_presenteado'])[:200]}"
                    for row_idx, row in produtos_sem_just.iterrows()
                ]

                prompt_justificativa = f'''
                    Você é a pessoa descrita no perfil a seguir e acabou de receber cada um dos itens listados como presente.
                    Perfil: {json.dumps(dados_json_sem_sexo, ensure_ascii=False)}.
                    Para cada produto da lista, escreva UMA frase de 100 a 200 caracteres reagindo ao presente como se você fosse essa pessoa.
                    Estrutura obrigatória: "Interjeição (Uau, Nossa, Que legal, etc) + (dizer o que é o produto) + "!" + "Eu achei bem" + (opinião do perfil sobre o produto) + "!"
                    FORMATO OBRIGATÓRIO: {{"justificativas": [{{"id": 0, "justificativa": "..."}}, ...]}}
                    IMPORTANTE: Retorne EXATAMENTE {len(produtos_sem_just)} itens, um para cada ID listado abaixo. Não deixe justificativa vazia.
                '''

                chaves_p4 = [k.strip() for k in [st.secrets.get("GROQ_KEY_1"), st.secrets.get("GROQ_KEY_2"), st.secrets.get("GROQ_KEY_3"), st.secrets.get("GROQ_KEY_4")] if k]
                modelos_p4 = [
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant"
                ]

                sucesso_p4 = False
                for modelo_p4 in modelos_p4:
                    if sucesso_p4:
                        break
                    chaves_tentativa_p4 = chaves_p4.copy()
                    while len(chaves_tentativa_p4) > 0:
                        chave_p4 = random.choice(chaves_tentativa_p4)
                        try:
                            client_p4 = Groq(api_key=chave_p4)
                            resp_p4 = client_p4.chat.completions.create(
                                model=modelo_p4,
                                messages=[
                                    {"role": "system", "content": "Você é um consultor que responde estritamente em JSON."},
                                    {"role": "user", "content": prompt_justificativa + "\n" + "\n".join(lista_p4)}
                                ],
                                response_format={"type": "json_object"}
                            )
                            if resp_p4.choices[0].finish_reason != "stop":
                                raise Exception("P4 não terminou corretamente")
                            dados_p4 = json.loads(resp_p4.choices[0].message.content)
                            just_p4_list = dados_p4.get("justificativas", [])
                            just_p4_map = {int(item['id']): item.get('justificativa', '') for item in just_p4_list}
                            for orig_idx, just_txt in just_p4_map.items():
                                if orig_idx in top_llm.index and just_txt.strip():
                                    top_llm.at[orig_idx, 'justificativa'] = just_txt
                            sucesso_p4 = True
                            break
                        except Exception:
                            chaves_tentativa_p4.remove(chave_p4)
                            continue
            # ---------------------------------------------------------------
            # Fim do Prompt 4
            # ---------------------------------------------------------------

            top_llm['score_final'] = (top_llm['score_cross_norm'] * 0.05) + (top_llm['score_llm'] * 0.95)
            top_llm.loc[top_llm['produto'].isin(st.session_state.liked_sessao), 'score_final'] += 2.0

            # Criar debug_json completo para o expander "COMO EU PENSEI"
            top_llm_sorted = top_llm.sort_values("score_final", ascending=False)
            debug_json = []
            for idx, row in top_llm_sorted.iterrows():
                if len(debug_json) >= 25: break
                debug_json.append({
                    "produto": row['produto'],
                    "score_encoder": float(row['score_cross_norm']),
                    "score_llm": float(row['score_llm']),
                    "score_final": float(row['score_final']),
                    "justificativa": row['justificativa'] if row['justificativa'] else ""
                })

            atualizar_progresso(90)
            top_llm['afinidade'] = top_llm['score_final']
            top_llm = top_llm.sort_values("afinidade", ascending=False)

            scores_cross_norm_all = np.zeros(len(df_ouro))
            for orig_idx, cross_val in zip(top_rec.index, top_rec['score_cross_norm'].values):
                if orig_idx < len(scores_cross_norm_all):
                    scores_cross_norm_all[orig_idx] = cross_val

            scores_llm_norm_all = np.zeros(len(df_ouro))
            for orig_idx, af_val in zip(top_llm.index, top_llm['afinidade'].values):
                if orig_idx < len(scores_llm_norm_all):
                    scores_llm_norm_all[orig_idx] = np.clip(af_val, 0, 1)

            MATCH_MINIMO = 0.55
            top5_df_grafico = top_llm.head(5)
            top5_orig_indices = list(top5_df_grafico.index)
            nomes_top5 = [str(n)[:10] for n in top5_df_grafico['produto'].values]

            scores_bi_all_v2 = np.zeros(len(df_ouro))
            for orig_idx, score_val in zip(df_f.index, df_f['score_bi'].values):
                scores_bi_all_v2[orig_idx] = np.clip(score_val, 0, None)
            s_max2 = scores_bi_all_v2.max()
            scores_bi_norm_v2 = scores_bi_all_v2 / (s_max2 + 1e-9)

            FRAMES_FASE_B = 5
            SLEEP_FASE_B = 0.08

            atualizar_progresso(92)
            for f in range(FRAMES_FASE_B + 1):
                t = f / FRAMES_FASE_B
                if t < 0.5:
                    a = ease(t / 0.5)
                    scores_frame = scores_bi_norm_v2 * (1 - a) + scores_cross_norm_all * a
                    titulo_frame = "Refinando com cross-encoder..."
                    mostrar_top5 = False
                else:
                    a = ease((t - 0.5) / 0.5)
                    scores_frame = scores_cross_norm_all * (1 - a) + scores_llm_norm_all * a
                    titulo_frame = "LLM selecionando os melhores presentes..."
                    mostrar_top5 = (t >= 0.95)

                fig_frame = render_grafico(
                    coords_todos=pca_produtos_coords,
                    star_pos=star_pos_pca,
                    scores_norm=scores_frame,
                    top5_indices=top5_orig_indices if mostrar_top5 else None,
                    nomes_top5=nomes_top5 if mostrar_top5 else None,
                    titulo=titulo_frame,
                )
                graph_placeholder.pyplot(fig_frame)
                plt.close(fig_frame)
                time.sleep(SLEEP_FASE_B)

            atualizar_progresso(100)
            fig_final = render_grafico(
                coords_todos=pca_produtos_coords,
                star_pos=star_pos_pca,
                scores_norm=scores_llm_norm_all,
                top5_indices=top5_orig_indices,
                nomes_top5=nomes_top5,
                titulo="Recomendações",
            )
            graph_placeholder.pyplot(fig_final)
            plt.close(fig_final)

            with col_grafico:
                st.markdown("<div style='text-align:center; font-size:0.8rem;'>★ Perfil do presenteado | ● Escala de Afinidade (Turbo)</div>", unsafe_allow_html=True)

            analise_completa = top_llm.head(25)[['produto', 'afinidade', 'score_llm', 'justificativa']].to_dict(orient='records')
            st.session_state.dados_ultima_busca = {
                "p1": res_tabela_raw,
                "p2": res_final,
                "p3": res_j_raw,
                "analise_top_30": analise_completa,
                "emb": v_busca,
                "clean_query": f"{necessidade}. {ultimo_input}",
                "dados_json": dados_json,
                "fb_html": fb_html,
                "fb_completo": tabela_feedback_completa,
                "fig_final": fig_final,
                "debug_json": debug_json,
                "modelo_usado": modelo_usado
            }
            ranking_threshold = top_llm[top_llm['afinidade'] >= MATCH_MINIMO].copy()
            st.session_state.ranking_atual = ranking_threshold
            st.rerun()

# --- EXIBIÇÃO DE RESULTADOS E BOTÕES DE FEEDBACK ---
if st.session_state.ranking_atual is not None:
    db = st.session_state.dados_ultima_busca
    with st.expander("🤓 COMO EU PENSEI...", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### MONTANDO O PERFIL...")
            st.json(db["dados_json"])
            st.info(db["p2"])
        with c2:
            st.markdown("#### COLHENDO FEEDBACKS ANTIGOS")
            if db.get("fb_completo"):
                fb_html_completo = render_fb_table_completa(
                    db["fb_completo"],
                    border_color=TEMA['border_color'],
                    thead_bg=TEMA['thead_bg'],
                    thead_color=TEMA['thead_color'],
                    cell_color=TEMA['cell_color']
                )
                st.markdown(fb_html_completo, unsafe_allow_html=True)
                st.markdown(
                    "<p style='margin-top:8px;'><small>" + str(len(db.get('fb_completo', []))) + " produtos com influência detectada.</small></p>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(db["fb_html"], unsafe_allow_html=True)

            with st.expander(f"REFINANDO RESULTADOS... (Modelo: {db.get('modelo_usado', 'N/A')})"):
                st.json(db["debug_json"])
        with c3:
            st.markdown("#### ESCOLHENDO PRODUTOS")
            st.pyplot(db["fig_final"])
            st.markdown("<div style='text-align:center; font-size:0.8rem;'>★ Perfil | ● Escala de Afinidade (Turbo)</div>", unsafe_allow_html=True)

    st.divider()

    if st.session_state.ranking_atual.empty:
        st.header("ACHEI! MINHA RECOMENDAÇÃO É:")
        st.markdown("### Nenhuma kkkkkk pode me dar mais detalhes? tô em dúvida!")
        st.info("🦗 *cri cri* vazio por aqui")
    else:
        vencedor = st.session_state.ranking_atual.iloc[0]
        match_score = min(vencedor['afinidade'], 1.0)
        n_recomendacoes = len(st.session_state.ranking_atual.head(10))

        if n_recomendacoes == 1:
            st.header("ACHEI! MINHA RECOMENDAÇÃO É:")
        else:
            st.header(f"ACHEI! MINHAS {n_recomendacoes} RECOMENDAÇÕES SÃO:")

        st.write(f"### Match Final: **{match_score*100:.1f}%**")
        st.progress(float(match_score))
        if vencedor['justificativa']:
            with st.chat_message("user", avatar="🗣️"):
                st.write(f"**Fala presenteado:** {vencedor['justificativa']}")
        st.success(f"VAI AMAR: **{vencedor['produto']}**")

        for i, row in st.session_state.ranking_atual.head(10).iterrows():
            c_match, c_produto, c_preco, c_link, c_justificativa, col_voto_sim, col_voto_nao = st.columns([1, 3, 1, 1.5, 4, 0.75, 0.75], vertical_alignment="center")
            with c_match: st.write(f"**{min(row['afinidade'], 1.0)*100:.1f}%**")
            c_produto.write(row['produto'])
            c_preco.write(f"R$ {row['preco']:.2f}")
            with c_link:
                url = str(row.get('link', '#'))
                label = "LINK AMAZON" if "amazon" in url.lower() else "LINK MELI" if ("mercadolivre" in url.lower() or "mlb." in url.lower()) else "LINK BOTICÁRIO" if "boticario" in url.lower() else "LINK RENNER"
                st.markdown(f"[{label}]({url})")
            c_justificativa.write(f"{row['justificativa'] if row['justificativa'] else ''}")

            voto_visual = "Sim" if (st.session_state.votos_temp.get(row['produto']) == "Sim" or row['produto'] in st.session_state.liked_sessao) else st.session_state.votos_temp.get(row['produto'])
            with col_voto_sim:
                if st.button(label="", icon="❤️", key=f"v_{i}", type="primary" if voto_visual == "Sim" else "secondary", use_container_width=True):
                    st.session_state.votos_temp[row['produto']] = "Sim"
                    st.rerun()
            with col_voto_nao:
                if st.button(label="", icon="❌", key=f"x_{i}", type="primary" if voto_visual == "Não" else "secondary", use_container_width=True):
                    st.session_state.votos_temp[row['produto']] = "Não"
                    st.rerun()
            st.divider()

if st.session_state.ranking_atual is not None:
    col_fb, col_rec = st.columns(2)
    with col_rec:
        if st.button("RECALCULAR SUGESTÕES", use_container_width=True, type="secondary"):
            for prod_nome, voto in st.session_state.votos_temp.items():
                if voto == "Não": st.session_state.banidos_sessao.add(prod_nome)
                if voto == "Sim": st.session_state.liked_sessao.add(prod_nome)
            st.session_state.ranking_atual = None
            st.session_state.votos_temp = {}
            st.session_state.houve_mudanca = True
            st.rerun()

    with col_fb:
        if st.session_state.feedback_enviado and not st.session_state.houve_mudanca:
            st.button("FEEDBACK ENVIADO!", disabled=True, use_container_width=True)
        else:
            fb_disabled = not bool(st.session_state.votos_temp)
            if st.button("ENVIAR FEEDBACK", disabled=fb_disabled, use_container_width=True, type="primary"):
                lista_fb = []
                for prod_nome, voto in st.session_state.votos_temp.items():
                    if voto == "Não": st.session_state.banidos_sessao.add(prod_nome)
                    if voto == "Sim": st.session_state.liked_sessao.add(prod_nome)
                    res_ouro = df_ouro[df_ouro['produto'] == prod_nome]
                    if not res_ouro.empty:
                        lista_fb.append({
                            "session_id": st.session_state.session_id,
                            "p1_json": db["p1"], "p2_texto": db["p2"],
                            "perfil_emb": db["emb"].tolist(), "produto": prod_nome,
                            "prod_emb": res_ouro['embedding_perfil'].values[0].tolist(), "voto": voto
                        })
                if lista_fb: salvar_lote_feedback(lista_fb)
                st.session_state.feedback_enviado = True
                st.session_state.houve_mudanca = False
                st.rerun()