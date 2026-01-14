import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap, SpectralEmbedding

st.set_page_config(page_title="PFE - Réduction de Dimension", layout="wide")

# --- TITRE ET CONFIGURATION ---
st.title("📊 Analyse et Visualisation de Données Multidimensionnelles")
st.sidebar.header("Paramètres de l'Application")

# --- 1. SÉLECTION ET PRÉPARATION DU DATASET ---
dataset_name = st.sidebar.selectbox("1. Choisir le Dataset", ("MNIST (Digits)", "Swiss Roll"))

def load_data(name):
    if name == "MNIST (Digits)":
        data = datasets.load_digits()
        X = data.data
        y = data.target
        return X, y, "2D"
    else:
        # Génération du Swiss Roll
        X, color = datasets.make_swiss_roll(n_samples=1500, noise=0.05)
        # CRUCIAL : La LDA ne supporte pas les labels continus (float).
        # On découpe le gradient de couleur en 10 classes distinctes (entiers).
        y_classes = np.digitize(color, np.linspace(color.min(), color.max(), 10))
        return X, y_classes, "3D"

X, y, default_type = load_data(dataset_name)

# --- 2. CHOIX DE LA MÉTHODE ---
method = st.sidebar.selectbox("2. Choisir la Méthode", 
                                ("PCA", "LDA", "Isomap", "Laplacian Eigenmaps"))

# --- 3. GESTION DYNAMIQUE DES DIMENSIONS (Évite l'erreur LDA) ---
num_classes = len(np.unique(y))
max_dim = 3 # Par défaut pour la visualisation

if method == "LDA":
    # La LDA est limitée mathématiquement par (nombre de classes - 1)
    max_lda = num_classes - 1
    max_dim = min(3, max_lda)
    st.sidebar.info(f"Note: Pour ce dataset, la LDA est limitée à {max_lda} dimensions.")

n_components = st.sidebar.slider("Dimensions finales", 2, max_dim, 2)

# Paramètres spécifiques pour Manifold Learning
params = {}
if method in ["Isomap", "Laplacian Eigenmaps"]:
    params['n_neighbors'] = st.sidebar.slider("Nombre de voisins (k)", 5, 50, 10)

# --- 4. EXÉCUTION DE L'ALGORITHME ---
@st.cache_data # Pour éviter de recalculer à chaque changement de slider
def run_reduction(X, y, method, n_comp, params):
    try:
        if method == "PCA":
            model = PCA(n_components=n_comp)
            return model.fit_transform(X)
        
        elif method == "LDA":
            model = LDA(n_components=n_comp)
            return model.fit_transform(X, y)
        
        elif method == "Isomap":
            model = Isomap(n_neighbors=params['n_neighbors'], n_components=n_comp)
            return model.fit_transform(X)
        
        elif method == "Laplacian Eigenmaps":
            model = SpectralEmbedding(n_neighbors=params['n_neighbors'], n_components=n_comp, affinity='nearest_neighbors')
            return model.fit_transform(X)
    except Exception as e:
        return f"Erreur : {str(e)}"

# Lancement du calcul
result = run_reduction(X, y, method, n_components, params)

# --- 5. AFFICHAGE ET VISUALISATION ---
if isinstance(result, str):
    st.error(result)
else:
    st.subheader(f"Projection {n_components}D via {method}")
    
    # Création du DataFrame pour Plotly
    cols = [f"Axe {i+1}" for i in range(n_components)]
    df = pd.DataFrame(result, columns=cols)
    df['Classe'] = y.astype(str) # Convertir en string pour une légende discrète

    if n_components == 2:
        fig = px.scatter(df, x="Axe 1", y="Axe 2", color="Classe", 
                         title=f"{method} sur {dataset_name} (2D)",
                         color_discrete_sequence=px.colors.qualitative.Spectral)
    else:
        fig = px.scatter_3d(df, x="Axe 1", y="Axe 2", z="Axe 3", color="Classe",
                            title=f"{method} sur {dataset_name} (3D)",
                            color_discrete_sequence=px.colors.qualitative.Spectral)
    
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")


