import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import Isomap, SpectralEmbedding

st.set_page_config(page_title="PFE - Réduction de Dimension", layout="wide")

st.title("📊 Analyse et Visualisation de Données Multidimensionnelles")
st.sidebar.header("Configuration")

# 1. Sélection du Dataset
dataset_name = st.sidebar.selectbox("Sélectionner le Dataset", ("MNIST (Digits)", "Swiss Roll"))

@st.cache_data
def get_dataset(name):
    if name == "MNIST (Digits)":
        data = datasets.load_digits()
        return data.data, data.target, "2D"
    else:
        # Swiss Roll generation
        X, color = datasets.make_swiss_roll(n_samples=1500, noise=0.05)
        # FIX 1: Discretize the continuous color into 10 classes for LDA compatibility
        # This creates integer labels (0-9) that LDA can use.
        y_discrete = np.digitize(color, np.linspace(color.min(), color.max(), 10))
        return X, y_discrete, "3D"

X, y, default_view = get_dataset(dataset_name)

# 2. Sélection de la Méthode
method = st.sidebar.selectbox("Méthode de Réduction", 
                                ("PCA", "LDA", "Isomap", "Laplacian Eigenmaps"))

# 3. Dynamic Slider Logic
# FIX 2: Prevent the slider from requesting more dimensions than LDA allows.
num_classes = len(np.unique(y))
max_dim_allowed = 3
if method == "LDA":
    max_dim_allowed = min(3, num_classes - 1)

n_components = st.sidebar.slider("Nombre de dimensions finales", 2, max_dim_allowed, 2)

# Parameters for Non-linear methods
params = {}
if method in ["Isomap", "Laplacian Eigenmaps"]:
    params['n_neighbors'] = st.sidebar.slider("Nombre de voisins (k)", 5, 50, 10)

# 4. Calculation
def apply_reduction(method, X, y, n_comp, params):
    try:
        if method == "PCA":
            model = PCA(n_components=n_comp)
            return model.fit_transform(X)
        elif method == "LDA":
            # Extra safety check for component constraints
            n_classes = len(np.unique(y))
            n_comp_lda = min(n_comp, n_classes - 1)
            model = LDA(n_components=n_comp_lda)
            return model.fit_transform(X, y)
        elif method == "Isomap":
            model = Isomap(n_neighbors=params['n_neighbors'], n_components=n_comp)
            return model.fit_transform(X)
        elif method == "Laplacian Eigenmaps":
            model = SpectralEmbedding(n_neighbors=params['n_neighbors'], n_components=n_comp)
            return model.fit_transform(X)
    except Exception as e:
        st.error(f"Erreur lors du calcul {method}: {e}")
        return None

X_reduced = apply_reduction(method, X, y, n_components, params)

# 5. Affichage
if X_reduced is not None:
    st.subheader(f"Résultat de la réduction via {method}")
    # Define columns based on actual components returned
    n_returned = X_reduced.shape[1]
    df = pd.DataFrame(X_reduced, columns=[f"Dim {i+1}" for i in range(n_returned)])
    df['label'] = y

    if n_returned == 2:
        fig = px.scatter(df, x="Dim 1", y="Dim 2", color="label", 
                         color_continuous_scale="Spectral", title=f"{method} 2D")
    else:
        fig = px.scatter_3d(df, x="Dim 1", y="Dim 2", z="Dim 3", color="label", 
                            color_continuous_scale="Spectral", title=f"{method} 3D")

    st.plotly_chart(fig, width=True)





