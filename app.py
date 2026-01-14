import streamlit as st
import pandas as pd
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

def get_dataset(name):
    if name == "MNIST (Digits)":
        data = datasets.load_digits()
        return data.data, data.target, "2D"
    else:
        X, color = datasets.make_swiss_roll(n_samples=1500, noise=0.05)
        # CORRECTION : Convertir le gradient continu en 10 classes discrètes pour la LDA
        y_discrete = np.digitize(color, np.linspace(color.min(), color.max(), 10))
        return X, y_discrete, "3D"

X, y, default_view = get_dataset(dataset_name)

# 2. Sélection de la Méthode
method = st.sidebar.selectbox("Méthode de Réduction", 
                                ("PCA", "LDA", "Isomap", "Laplacian Eigenmaps"))

n_components = st.sidebar.slider("Nombre de dimensions finales", 2, 3, 2)

# Paramètres spécifiques aux méthodes non-linéaires
params = {}
if method in ["Isomap", "Laplacian Eigenmaps"]:
    params['n_neighbors'] = st.sidebar.slider("Nombre de voisins (k)", 5, 50, 10)

# 3. Calcul
def apply_reduction(method, X, y, n_comp, params):
    if method == "PCA":
        model = PCA(n_components=n_comp)
        return model.fit_transform(X)
    elif method == "LDA":
        # La LDA est limitée par le nombre de classes - 1
        model = LDA(n_components=min(n_comp, len(set(y))-1))
        return model.fit_transform(X, y)
    elif method == "Isomap":
        model = Isomap(n_neighbors=params['n_neighbors'], n_components=n_comp)
        return model.fit_transform(X)
    elif method == "Laplacian Eigenmaps":
        model = SpectralEmbedding(n_neighbors=params['n_neighbors'], n_components=n_comp)
        return model.fit_transform(X)

X_reduced = apply_reduction(method, X, y, n_components, params)

# 4. Affichage
st.subheader(f"Résultat de la réduction via {method}")
df = pd.DataFrame(X_reduced, columns=[f"Dim {i+1}" for i in range(n_components)])
df['label'] = y

if n_components == 2:
    fig = px.scatter(df, x="Dim 1", y="Dim 2", color="label", color_continuous_scale="Spectral")
else:
    fig = px.scatter_3d(df, x="Dim 1", y="Dim 2", z="Dim 3", color="label", color_continuous_scale="Spectral")


st.plotly_chart(fig, use_container_width=True)
