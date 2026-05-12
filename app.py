"""
Interfaz visual para el Clasificador de Familias de Proteínas.
Soporta embeddings ESM-2 + features manuales.
Ejecutar con: streamlit run app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import os

AMINOACIDOS = 'ACDEFGHIKLMNPQRSTVWY'

PROPIEDADES_AA = {
    'A': {'hydro': 1.8, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'C': {'hydro': 2.5, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 0, 'aromatic': 0},
    'D': {'hydro': -3.5, 'polar': 1, 'charge': -1, 'size': 0, 'flex': 1, 'aromatic': 0},
    'E': {'hydro': -3.5, 'polar': 1, 'charge': -1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'F': {'hydro': 2.8, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
    'G': {'hydro': -0.4, 'polar': 0, 'charge': 0, 'size': -1, 'flex': 1, 'aromatic': 0},
    'H': {'hydro': -3.2, 'polar': 1, 'charge': 0.5, 'size': 1, 'flex': 0, 'aromatic': 1},
    'I': {'hydro': 4.5, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'K': {'hydro': -3.9, 'polar': 1, 'charge': 1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'L': {'hydro': 3.8, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'M': {'hydro': 1.9, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 0},
    'N': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'P': {'hydro': -1.6, 'polar': 0, 'charge': 0, 'size': 0, 'flex': -1, 'aromatic': 0},
    'Q': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'size': 1, 'flex': 1, 'aromatic': 0},
    'R': {'hydro': -4.5, 'polar': 1, 'charge': 1, 'size': 1, 'flex': 1, 'aromatic': 0},
    'S': {'hydro': -0.8, 'polar': 1, 'charge': 0, 'size': -1, 'flex': 1, 'aromatic': 0},
    'T': {'hydro': -0.7, 'polar': 1, 'charge': 0, 'size': 0, 'flex': 1, 'aromatic': 0},
    'V': {'hydro': 4.2, 'polar': 0, 'charge': 0, 'size': 0, 'flex': 0, 'aromatic': 0},
    'W': {'hydro': -0.9, 'polar': 0, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
    'Y': {'hydro': -1.3, 'polar': 1, 'charge': 0, 'size': 1, 'flex': 0, 'aromatic': 1},
}

GRUPOS_AA = {
    'nonpolar': 'AILMFVW',
    'polar_uncharged': 'NQSTY',
    'positively_charged': 'KRH',
    'negatively_charged': 'DE',
    'aromatic': 'FWY',
    'aliphatic': 'ILV',
    'tiny': 'AGS',
    'special': 'CP',
}


def extraer_caracteristicas_manuales(secuencia):
    if not secuencia or len(secuencia) == 0:
        return [0.0] * (20 + 400 + 7 + 3 + 6 + 20 + 3)

    seq = secuencia.upper()
    n = len(seq)
    feats = []

    composicion = [seq.count(aa) / n for aa in AMINOACIDOS]
    feats.extend(composicion)

    dipeptidos = []
    for aa1 in AMINOACIDOS:
        for aa2 in AMINOACIDOS:
            dip = aa1 + aa2
            count = seq.count(dip)
            dipeptidos.append(count / max(n - 1, 1))
    feats.extend(dipeptidos)

    for prop_name in ['hydro', 'polar', 'charge', 'size', 'flex', 'aromatic']:
        valores = [PROPIEDADES_AA.get(aa, {}).get(prop_name, 0) for aa in seq]
        if valores:
            feats.append(np.mean(valores))
            feats.append(np.std(valores))
            feats.append(np.min(valores))
            if prop_name == 'hydro':
                ventana = 9
                hydro_mean_vals = []
                for i in range(max(1, n - ventana + 1)):
                    window_vals = valores[i:i + ventana]
                    hydro_mean_vals.append(np.mean(window_vals))
                if hydro_mean_vals:
                    feats.append(np.max(hydro_mean_vals))
                    feats.append(np.min(hydro_mean_vals))
                    feats.append(np.max(hydro_mean_vals) - np.min(hydro_mean_vals))
                else:
                    feats.extend([0, 0, 0])
        else:
            feats.extend([0, 0, 0, 0, 0, 0])

    feats.extend([
        n,
        np.log1p(n),
        (seq.count('K') + seq.count('R') + seq.count('H')) / n,
        (seq.count('D') + seq.count('E')) / n,
        seq.count('C') / n,
        seq.count('P') / n,
    ])

    for grupo_nombre, grupo_aas in GRUPOS_AA.items():
        freq = sum(seq.count(aa) for aa in grupo_aas) / n
        feats.append(freq)

    tercio = n // 3
    for i in range(3):
        inicio = i * tercio
        fin = (i + 1) * tercio if i < 2 else n
        subseq = seq[inicio:fin]
        if len(subseq) > 0:
            for aa in AMINOACIDOS:
                feats.append(subseq.count(aa) / len(subseq))
        else:
            feats.extend([0.0] * 20)

    return feats


st.set_page_config(
    page_title="Clasificador de Proteínas",
    page_icon="🧬",
    layout="wide"
)


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelos/clasificador_proteinas.joblib")
    encoder = joblib.load("modelos/label_encoder.joblib")

    info = {}
    info_path = "modelos/feature_info.json"
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)

    scaler_manual = None
    scaler_emb = None
    embedder = None
    use_emb = info.get('use_embeddings', False)
    esm_model = info.get('esm_model', 'esm2_t6_8M_UR50D')

    if os.path.exists("modelos/scaler_manual.joblib"):
        scaler_manual = joblib.load("modelos/scaler_manual.joblib")
    if os.path.exists("modelos/scaler_emb.joblib"):
        scaler_emb = joblib.load("modelos/scaler_emb.joblib")

    if use_emb:
        try:
            import importlib
            emb_module = importlib.import_module('embeddings')
            ProteinEmbedder = emb_module.ProteinEmbedder
            embedder = ProteinEmbedder(model_name=esm_model)
        except Exception:
            use_emb = False

    return modelo, encoder, info, scaler_manual, scaler_emb, embedder, use_emb


modelo, encoder, model_info, scaler_manual, scaler_emb, embedder, use_embeddings = cargar_modelo()


def predecir_familia(secuencia):
    manual_feats = np.array([extraer_caracteristicas_manuales(secuencia)], dtype=np.float32)

    if scaler_manual is not None:
        manual_feats = scaler_manual.transform(manual_feats)

    if use_embeddings and embedder is not None:
        emb = embedder.embed_sequence(secuencia).reshape(1, -1)
        if scaler_emb is not None:
            emb = scaler_emb.transform(emb)
        features = np.hstack([emb, manual_feats])
    else:
        features = manual_feats

    prediccion_idx = modelo.predict(features)[0]
    probabilidades = modelo.predict_proba(features)[0]

    familia = encoder.inverse_transform([prediccion_idx])[0]

    top_indices = np.argsort(probabilidades)[::-1][:5]
    top_predicciones = [
        (encoder.inverse_transform([i])[0], float(probabilidades[i]))
        for i in top_indices
    ]

    return familia, top_predicciones


st.title("🧬 Clasificador de Familias de Proteínas")
mode_str = "ESM-2 + Features manuales" if use_embeddings else "Features manuales"
st.markdown(f"""
**IA para clasificación de proteínas** | Modo: **{mode_str}**
""")

with st.sidebar:
    st.header("ℹ️ Información")
    st.markdown(f"""
    ### Modelo actual
    - Familias: {len(encoder.classes_)}
    - Precisión: {model_info.get('accuracy', 'N/A')}
    - Features: {model_info.get('num_features', 'N/A')}
    - Embeddings: {'Sí (' + str(model_info.get('esm_model', '')) + ')' if use_embeddings else 'No'}

    ### Aminoácidos válidos
    `A C D E F G H I K L M N P Q R S T V W Y`
    """)

    st.header("📊 Familias Disponibles")
    familias = encoder.classes_
    search = st.text_input("Buscar familia:", "")
    filtered = [f for f in familias if search.upper() in f.upper()] if search else familias
    for fam in filtered[:50]:
        st.text(f"• {fam}")
    if len(filtered) > 50:
        st.text(f"... y {len(filtered) - 50} más")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Ingresa tu secuencia")

    secuencia_input = st.text_area(
        "Secuencia de aminoácidos:",
        height=150,
        placeholder="Ej: MKWVTFISLLFLFSSAYSRG...",
        help="Ingresa la secuencia completa de la proteína"
    )

    if st.button("🔬 Predecir Familia", type="primary", use_container_width=True):
        if secuencia_input.strip():
            secuencia_limpia = ''.join(
                c for c in secuencia_input.upper()
                if c in 'ACDEFGHIKLMNPQRSTVWY'
            )

            if len(secuencia_limpia) < 10:
                st.error("⚠️ La secuencia es muy corta (mínimo 10 aminoácidos)")
            else:
                with st.spinner("Analizando secuencia..."):
                    familia, top = predecir_familia(secuencia_limpia)

                st.success("✅ Predicción completada")

                st.markdown("### 🎯 Resultado")
                st.metric(label="Familia Predicha", value=familia)

                st.markdown("### 📊 Probabilidades")
                df_probs = pd.DataFrame({
                    "Familia": [t[0] for t in top],
                    "Probabilidad": [t[1] for t in top]
                })

                st.bar_chart(df_probs.set_index("Familia"), use_container_width=True)
                st.dataframe(
                    df_probs.style.format({"Probabilidad": "{:.2%}"}),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning("⚠️ Ingresa una secuencia para predecir")

with col2:
    st.subheader("📈 Estadísticas")

    if secuencia_input.strip():
        secuencia_limpia = ''.join(
            c for c in secuencia_input.upper()
            if c in 'ACDEFGHIKLMNPQRSTVWY'
        )

        if secuencia_limpia:
            composicion = {
                aa: secuencia_limpia.count(aa) / len(secuencia_limpia) * 100
                for aa in AMINOACIDOS
            }

            st.metric("Longitud", f"{len(secuencia_limpia)} aa")
            st.metric("Peso molecular aprox.", f"{len(secuencia_limpia) * 110:.0f} Da")

            top_aa = sorted(composicion.items(), key=lambda x: x[1], reverse=True)[:5]
            nombres_aa = {
                'A': 'Alanina', 'C': 'Cisteína', 'D': 'Ác. Aspártico',
                'E': 'Ác. Glutámico', 'F': 'Fenilalanina', 'G': 'Glicina',
                'H': 'Histidina', 'I': 'Isoleucina', 'K': 'Lisina',
                'L': 'Leucina', 'M': 'Metionina', 'N': 'Asparagina',
                'P': 'Prolina', 'Q': 'Glutamina', 'R': 'Arginina',
                'S': 'Serina', 'T': 'Treonina', 'V': 'Valina',
                'W': 'Triptófano', 'Y': 'Tirosina'
            }

            st.markdown("**Top 5 aminoácidos:**")
            for aa, pct in top_aa:
                st.text(f"{aa} ({nombres_aa.get(aa, '')}): {pct:.1f}%")

            df_comp = pd.DataFrame({
                "Aminoácido": list(AMINOACIDOS),
                "Composición (%)": [composicion[aa] for aa in AMINOACIDOS]
            })
            st.bar_chart(df_comp.set_index("Aminoácido"), use_container_width=True)
    else:
        st.info("Ingresa una secuencia para ver estadísticas")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<small>Clasificador de Proteínas v2.0 | ESM-2 + Features fisicoquímicos | Dataset: Pfam</small>
</div>
""", unsafe_allow_html=True)
