"""
Módulo de embeddings de proteínas usando ESM-2 de Meta via HuggingFace.
Genera representaciones vectoriales de secuencias de aminoácidos.
Funciona en CPU (lento) o GPU (rápido) - sin limitaciones de VRAM local
porque descarga el modelo y lo corre en la máquina local o en cloud.
"""

import numpy as np
import os

try:
    import torch
except ImportError:
    torch = None

ESM_MODELS = {
    'esm2_t6_8M_UR50D': {
        'hub_id': 'facebook/esm2_t6_8M_UR50D',
        'emb_dim': 320,
        'layers': 6,
        'description': 'ESM-2 pequeño (8M params, 320-dim) - Rápido, buena precisión',
    },
    'esm2_t12_35M_UR50D': {
        'hub_id': 'facebook/esm2_t12_35M_UR50D',
        'emb_dim': 480,
        'layers': 12,
        'description': 'ESM-2 mediano (35M params, 480-dim) - Balance velocidad/precisión',
    },
    'esm2_t30_150M_UR50D': {
        'hub_id': 'facebook/esm2_t30_150M_UR50D',
        'emb_dim': 640,
        'layers': 30,
        'description': 'ESM-2 grande (150M params, 640-dim) - Alta precisión',
    },
    'esm2_t33_650M_UR50D': {
        'hub_id': 'facebook/esm2_t33_650M_UR50D',
        'emb_dim': 1280,
        'layers': 33,
        'description': 'ESM-2 XL (650M params, 1280-dim) - Mejor precisión, requiere GPU',
    },
}

DEFAULT_MODEL = 'esm2_t6_8M_UR50D'

class ProteinEmbedder:
    def __init__(self, model_name=None, device=None):
        if torch is None:
            raise ImportError("PyTorch no está instalado. Instala con: pip install torch")

        if model_name is None:
            model_name = DEFAULT_MODEL

        if model_name not in ESM_MODELS:
            raise ValueError(f"Modelo {model_name} no encontrado. Opciones: {list(SM_MODELS.keys())}")

        self.model_name = model_name
        self.config = ESM_MODELS[model_name]

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Cargando modelo {self.model_name} ({self.config['description']})...")
        print(f"Dispositivo: {self.device}")

        try:
            from transformers import EsmModel, EsmTokenizer
        except ImportError:
            print("Instalando transformers...")
            os.system("pip install transformers")
            from transformers import EsmModel, EsmTokenizer

        self.tokenizer = EsmTokenizer.from_pretrained(self.config['hub_id'])
        self.model = EsmModel.from_pretrained(self.config['hub_id'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Modelo cargado. Dimensión de embedding: {self.config['emb_dim']}")

    def embed_sequence(self, sequence):
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences, max_length=1024, batch_size=8):
        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch = [s[:max_length - 2] for s in batch]

            token_inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            token_inputs = {k: v.to(self.device) for k, v in token_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**token_inputs)
                last_hidden = outputs.last_hidden_state

            attention_mask = token_inputs['attention_mask'].unsqueeze(-1)
            sum_embeddings = (last_hidden * attention_mask).sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / lengths

            all_embeddings.append(mean_embeddings.cpu().numpy())

            if (i + batch_size) % 100 == 0 or i + batch_size >= len(sequences):
                print(f"  Embeddings: {min(i + batch_size, len(sequences))}/{len(sequences)} secuencias")

        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self):
        return self.config['emb_dim']


def generate_embeddings_cache(sequences, labels, output_dir="modelos/embeddings_cache",
                               model_name=None, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)

    embedder = ProteinEmbedder(model_name=model_name)

    print(f"Generando embeddings para {len(sequences)} secuencias...")
    embeddings = embedder.embed_batch(sequences, batch_size=batch_size)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    metadata = {
        'model_name': embedder.model_name,
        'embedding_dim': embedder.embedding_dim,
        'num_sequences': len(sequences),
        'device': str(embedder.device),
    }

    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Embeddings guardados en {output_dir}/")
    print(f"Shape: {embeddings.shape}")

    return embeddings


def load_embeddings_cache(cache_dir="modelos/embeddings_cache"):
    embeddings = np.load(os.path.join(cache_dir, "embeddings.npy"))
    labels = np.load(os.path.join(cache_dir, "labels.npy"), allow_pickle=True)

    import json
    with open(os.path.join(cache_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    print(f"Embeddings cargados: {embeddings.shape} desde cache")
    return embeddings, labels, metadata


def has_embeddings_cache(cache_dir="modelos/embeddings_cache"):
    return os.path.exists(os.path.join(cache_dir, "embeddings.npy"))
