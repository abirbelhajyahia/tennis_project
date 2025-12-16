import json
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Constantes ---
DATA_DIR = "data/per_point_v2"
MODEL_PATH = "supervised_model.joblib"

# --- Utilitaires de Traitement de Données ---

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Charge les données JSON d'un point et les prétraite.
    Calcule la vitesse et l'accélération.
    """
    with open(file_path, 'r') as f:
        raw_data = json.load(f)

    # Convertir les données en DataFrame
    records = []
    for frame_str, attrs in raw_data.items():
        frame = int(frame_str)
        records.append({
            'frame': frame,
            'x': attrs.get('x'),
            'y': attrs.get('y'),
            'visible': attrs.get('visible', False),
            'action': attrs.get('action', 'air'),
        })

    df = pd.DataFrame(records).sort_values('frame').reset_index(drop=True)
    
    # Remplacer les valeurs manquantes (quand la balle n'est pas visible)
    # Pour la détection non supervisée, nous nous concentrons sur les moments où la balle est visible.
    # Pour l'entraînement supervisé, nous pourrions avoir besoin d'une imputation plus sophistiquée,
    # mais pour l'instant, nous allons utiliser la dernière position connue pour les frames non visibles
    # et marquer les frames non visibles.
    df['x_interp'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y_interp'] = df['y'].interpolate(method='linear', limit_direction='both')
    
    # Calculer la vitesse (différence première)
    # La vitesse est calculée par rapport à la frame précédente.
    df['vx'] = df['x_interp'].diff().fillna(0)
    df['vy'] = df['y_interp'].diff().fillna(0)
    
    # Calculer l'accélération (différence seconde)
    df['ax'] = df['vx'].diff().fillna(0)
    df['ay'] = df['vy'].diff().fillna(0)
    
    # Calculer la vitesse et l'accélération totales
    df['v_total'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['a_total'] = np.sqrt(df['ax']**2 + df['ay']**2)
    
    return df

def process_all_data(data_dir: str = DATA_DIR) -> List[pd.DataFrame]:
    """Charge et prétraite tous les fichiers JSON du répertoire."""
    all_dfs = []
    for file_path in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            df = load_and_preprocess_data(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file_path}: {e}")
    return all_dfs

# --- Méthode 1 : Détection Non Supervisée (Physique) ---

def unsupervised_hit_bounce_detection(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Détecte les hits et les bounces en utilisant une approche non supervisée
    basée sur l'analyse des changements de vitesse et d'accélération.
    """
    df = load_and_preprocess_data(file_path)
    
    # Détection basée sur l'accélération verticale (ay)
    # Un hit ou un bounce provoque un pic d'accélération (changement de direction/vitesse).
    
    # 1. Calculer l'accélération verticale absolue
    df['abs_ay'] = df['ay'].abs()
    
    # 2. Définir un seuil pour les pics d'accélération
    # Le seuil doit être robuste. On peut utiliser une méthode statistique,
    # par exemple, un multiple de l'écart-type de l'accélération.
    # Pour un jeu de données simulé, un seuil fixe peut être nécessaire.
    # Dans un cas réel, on pourrait utiliser 3 ou 4 * std(ay)
    
    # Calculer l'écart-type de l'accélération verticale pour les frames "air" (bruit de fond)
    # Comme nous n'avons pas de données réelles, nous allons utiliser un seuil arbitraire
    # basé sur l'observation de la simulation.
    
    # Seuil arbitraire basé sur la physique: un changement de vitesse significatif
    # Le seuil doit être ajusté en fonction des données réelles.
    # Ici, nous allons utiliser un seuil basé sur la déviation standard de l'accélération totale.
    
    # Pour l'exemple, nous allons utiliser un seuil simple sur l'accélération totale.
    # L'accélération est généralement faible en l'air (gravité), mais élevée lors d'un contact.
    
    # Calculer la moyenne et l'écart-type de l'accélération totale
    mean_a = df['a_total'].mean()
    std_a = df['a_total'].std()
    
    # Seuil: 3 écarts-types au-dessus de la moyenne
    threshold = mean_a + 3 * std_a
    
    # 3. Identifier les pics d'accélération
    df['is_candidate'] = (df['a_total'] > threshold) & df['visible']
    
    # 4. Affiner la détection (Hit vs Bounce)
    # Un bounce se produit généralement lorsque la balle remonte (vy passe de positif à négatif)
    # Un hit peut se produire n'importe où, mais est un changement de vitesse brutal.
    
    # Pour simplifier, nous allons considérer tous les pics comme des événements (hit ou bounce)
    # et les différencier par la direction de la balle après l'événement.
    
    df['pred_action'] = 'air'
    
    # Parcourir les candidats
    for i in df[df['is_candidate']].index:
        # S'assurer que le pic est isolé (pas de détection dans les frames adjacentes)
        if i > 0 and df.loc[i-1, 'is_candidate']:
            continue
        
        # Un hit ou un bounce est un événement ponctuel.
        # On peut différencier un bounce d'un hit par la position Y (proche du sol)
        # et le changement de signe de la vitesse verticale (vy).
        
        # Bounce: vy passe de positif (vers le bas, y augmente) à négatif (vers le haut, y diminue)
        # Un hit est un changement de vitesse brutal en l'air.
        
        # Pour l'instant, nous allons simplement marquer le pic comme "hit" ou "bounce"
        # en fonction de la position Y (arbitraire, y > 800 est proche du sol dans un cadre 1080p)
        
        if df.loc[i, 'y_interp'] > 800: # Proche du sol (hypothèse)
            df.loc[i, 'pred_action'] = 'bounce'
        else:
            df.loc[i, 'pred_action'] = 'hit'
            
    # 5. Formater la sortie
    output_data = {}
    for _, row in df.iterrows():
        frame_str = str(row['frame'])
        output_data[frame_str] = {
            "x": int(row['x']) if not pd.isna(row['x']) else None,
            "y": int(row['y']) if not pd.isna(row['y']) else None,
            "visible": bool(row['visible']),
            "action": row['action'], # Ground Truth
            "pred_action": row['pred_action'] # Prédiction
        }
        
    return output_data

# --- Méthode 2 : Détection Supervisée (Machine Learning) ---

def train_supervised_model(data_dir: str = DATA_DIR, model_path: str = MODEL_PATH) -> None:
    """
    Charge toutes les données, extrait les caractéristiques, entraîne un modèle
    et le sauvegarde.
    """
    all_dfs = process_all_data(data_dir)
    
    if not all_dfs:
        print("Aucune donnée à entraîner. Le modèle ne sera pas entraîné.")
        return

    # Concaténer tous les points en un seul DataFrame
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Création des caractéristiques (Features)
    # Nous utilisons les positions, vitesses, accélérations et leurs valeurs décalées
    # pour capturer la dynamique temporelle.
    
    features = ['x_interp', 'y_interp', 'vx', 'vy', 'ax', 'ay', 'v_total', 'a_total']
    
    # Ajouter des caractéristiques temporelles (lagged features)
    for lag in range(1, 4): # Utiliser les 3 frames précédentes
        for feature in ['x_interp', 'y_interp', 'vx', 'vy', 'ax', 'ay']:
            full_df[f'{feature}_lag_{lag}'] = full_df[feature].shift(lag).fillna(0)
            features.append(f'{feature}_lag_{lag}')
            
    # Filtrer les lignes avec des valeurs NaN (après le shift)
    full_df.dropna(subset=features, inplace=True)
    
    # Encodage de la variable cible (action)
    # action: 'air', 'hit', 'bounce'
    target_mapping = {'air': 0, 'hit': 1, 'bounce': 2}
    full_df['target'] = full_df['action'].map(target_mapping)
    
    # Préparation des données pour l'entraînement
    X = full_df[features]
    y = full_df['target']
    
    # Séparation des données (entraînement/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entraînement du modèle (Random Forest est un bon choix pour commencer)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Évaluation (optionnel, mais recommandé)
    y_pred = model.predict(X_test)
    print("\n--- Rapport de Classification (Supervisé) ---")
    print(classification_report(y_test, y_pred, target_names=['air', 'hit', 'bounce']))
    
    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"Modèle supervisé entraîné et sauvegardé à {model_path}")

def supervised_hit_bounce_detection(file_path: str, model_path: str = MODEL_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Détecte les hits et les bounces en utilisant le modèle supervisé entraîné.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Erreur: Modèle non trouvé à {model_path}. Veuillez l'entraîner d'abord.")
        return {}
        
    df = load_and_preprocess_data(file_path)
    
    # Préparation des caractéristiques pour la prédiction
    features = ['x_interp', 'y_interp', 'vx', 'vy', 'ax', 'ay', 'v_total', 'a_total']
    for lag in range(1, 4):
        for feature in ['x_interp', 'y_interp', 'vx', 'vy', 'ax', 'ay']:
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag).fillna(0)
            features.append(f'{feature}_lag_{lag}')
            
    # Filtrer les lignes avec des valeurs NaN (après le shift)
    # Les premières lignes seront ignorées car elles n'ont pas assez de données de lag.
    df_predict = df.dropna(subset=features).copy()
    
    # Prédiction
    X_predict = df_predict[features]
    y_pred = model.predict(X_predict)
    
    # Mappage inverse
    label_mapping = {0: 'air', 1: 'hit', 2: 'bounce'}
    df_predict['pred_action'] = [label_mapping[p] for p in y_pred]
    
    # Fusionner les prédictions avec le DataFrame original
    df = df.merge(df_predict[['frame', 'pred_action']], on='frame', how='left')
    df['pred_action'] = df['pred_action'].fillna('air') # Les frames sans prédiction sont 'air'
    
    # Formater la sortie
    output_data = {}
    for _, row in df.iterrows():
        frame_str = str(row['frame'])
        output_data[frame_str] = {
            "x": int(row['x']) if not pd.isna(row['x']) else None,
            "y": int(row['y']) if not pd.isna(row['y']) else None,
            "visible": bool(row['visible']),
            "action": row['action'],
            "pred_action": row['pred_action']
        }
        
    return output_data

# --- Fonction Principale ---

def main():
    """
    Fonction principale pour exécuter les détections sur tous les fichiers de données.
    """
    # 1. Entraîner le modèle supervisé (nécessaire avant la détection supervisée)
    print("--- Étape 1: Entraînement du modèle supervisé ---")
    train_supervised_model()
    
    # 2. Exécuter les détections sur un échantillon de données
    print("\n--- Étape 2: Exécution des détections ---")
    
    # Utiliser le premier fichier de données simulé pour la démonstration
    sample_file = os.path.join(DATA_DIR, "ball_data_1.json")
    
    if not os.path.exists(sample_file):
        print(f"Fichier d'échantillon non trouvé: {sample_file}. Veuillez exécuter generate_mock_data.py.")
        return

    # Détection Non Supervisée
    print(f"\n-> Détection Non Supervisée pour {sample_file}")
    unsupervised_result = unsupervised_hit_bounce_detection(sample_file)
    
    # Afficher les événements détectés (hit ou bounce)
    unsupervised_events = {k: v for k, v in unsupervised_result.items() if v['pred_action'] in ['hit', 'bounce']}
    print(f"Événements non supervisés détectés: {unsupervised_events}")
    
    # Détection Supervisée
    print(f"\n-> Détection Supervisée pour {sample_file}")
    supervised_result = supervised_hit_bounce_detection(sample_file)
    
    # Afficher les événements détectés (hit ou bounce)
    supervised_events = {k: v for k, v in supervised_result.items() if v['pred_action'] in ['hit', 'bounce']}
    print(f"Événements supervisés détectés: {supervised_events}")
    
    # 3. Sauvegarder les résultats enrichis (optionnel, mais utile pour la vérification)
    # Exemple de sauvegarde du résultat non supervisé
    output_file = "unsupervised_result_sample.json"
    with open(output_file, 'w') as f:
        json.dump(unsupervised_result, f, indent=4)
    print(f"\nRésultat non supervisé sauvegardé dans {output_file}")
    
    # Exemple de sauvegarde du résultat supervisé
    output_file = "supervised_result_sample.json"
    with open(output_file, 'w') as f:
        json.dump(supervised_result, f, indent=4)
    print(f"Résultat supervisé sauvegardé dans {output_file}")


if __name__ == "__main__":
    main()
