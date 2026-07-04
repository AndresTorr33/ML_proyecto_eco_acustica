import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# Cargar y estandarizar datos (usando ruta relativa desde src/)
df = pd.read_csv('../data/raw/eco_acoustic_train.csv')
X = df.loc[:, 'mel_0':'mel_63']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Entrenar los modelos definitivos con los hiperparámetros óptimos
# Mejor GMM (K=3)
mejor_gmm = GaussianMixture(n_components=3, random_state=42)
mejor_gmm.fit(X_scaled)

# Mejor DBSCAN (eps=6.0, min_samples=3)
mejor_dbscan = DBSCAN(eps=6.0, min_samples=3)
mejor_dbscan.fit(X_scaled) # Nota: DBSCAN no tiene un método .predict() para datos nuevos, 
                           # pero es bueno guardarlo si necesitan extraer las etiquetas (labels_) de entrenamiento.

# 3. Exportar los modelos y el escalador a la carpeta models_saved/
joblib.dump(mejor_gmm, '../models_saved/gmm_k3.pkl')
joblib.dump(mejor_dbscan, '../models_saved/dbscan_optimo.pkl')
joblib.dump(scaler, '../models_saved/scaler.pkl') # Vital guardar el scaler para que Streamlit normalice los datos de test igual que en train

print("¡Modelos y escalador guardados con éxito en models_saved/!")