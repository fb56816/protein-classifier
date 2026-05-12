"""
Script para predecir la familia de nuevas secuencias de proteínas.
Usa embeddings ESM-2 + features manuales (igual que el entrenamiento).
"""

import joblib
import numpy as np
import os
import json

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


modelo = joblib.load("modelos/clasificador_proteinas.joblib")
label_encoder = joblib.load("modelos/label_encoder.joblib")

scaler_manual = None
scaler_emb = None
embedder = None
use_embeddings = False

feature_info_path = "modelos/feature_info.json"
if os.path.exists(feature_info_path):
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)
    use_embeddings = feature_info.get('use_embeddings', False)
    esm_model = feature_info.get('esm_model', 'esm2_t6_8M_UR50D')

    if os.path.exists("modelos/scaler_manual.joblib"):
        scaler_manual = joblib.load("modelos/scaler_manual.joblib")
    if os.path.exists("modelos/scaler_emb.joblib"):
        scaler_emb = joblib.load("modelos/scaler_emb.joblib")

    if use_embeddings:
        try:
            from embeddings import ProteinEmbedder
            embedder = ProteinEmbedder(model_name=esm_model)
            print(f"Embeddings ESM-2 activos ({esm_model})")
        except Exception as e:
            print(f"Advertencia: no se pudo cargar embedder: {e}")
            use_embeddings = False


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

    familia = label_encoder.inverse_transform([prediccion_idx])[0]

    top_indices = np.argsort(probabilidades)[::-1][:5]
    top_predicciones = [
        (label_encoder.inverse_transform([i])[0], float(probabilidades[i]))
        for i in top_indices
    ]

    return familia, top_predicciones


if __name__ == "__main__":
    print("=== Clasificador de Familias de Proteínas ===")
    print(f"Modelo cargado: {len(label_encoder.classes_)} familias")
    if use_embeddings:
        print(f"Modo: Embeddings ESM-2 + Features manuales")
    else:
        print(f"Modo: Solo features manuales")
    print()

    secuencias_ejemplo = [
        "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQC",
        "MTEITAAMVKELRESTGAGMMDCKNALSETNGDGKKIYVKDRDVGKVLHEQVKG",
        "MRSLGYQEKAIAALGKGDVIVVTGPTIANVGHLKQTGVQKAFDKVYRIDRDLK",
    ]

    for seq in secuencias_ejemplo:
        familia, top = predecir_familia(seq)
        print(f"Secuencia: {seq[:40]}...")
        print(f"  Familia predicha: {familia}")
        print(f"  Top 3:")
        for fam, prob in top[:3]:
            print(f"    {fam}: {prob:.2%}")
        print()

    print("=== Modo interactivo ===")
    print("Ingresa una secuencia de proteína (o 'salir' para terminar):\n")

    while True:
        secuencia = input("> ").strip().upper()
        if secuencia.lower() in ['salir', 'exit', 'q']:
            break
        if not secuencia:
            continue

        familia, top = predecir_familia(secuencia)
        print(f"\nFamilia predicha: {familia}")
        print("Top 5:")
        for fam, prob in top:
            print(f"  {fam}: {prob:.2%}")
        print()
