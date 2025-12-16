# Tennis Hits & Bounces Detection - Roland Garros 2025 Final

Ce projet implémente deux méthodes de détection des **hits** (frappes) et des **bounces** (rebonds) de balle de tennis à partir de données de suivi de balle (x, y) extraites de la finale de Roland Garros 2025.

## Structure du Projet

```
.
├── data/
│   └── per_point_v2/
│       └── ball_data_*.json  (Données de suivi de balle par point)
├── main.py                   (Contient les deux fonctions de détection et la logique d'entraînement)
├── supervised_model.joblib   (Modèle supervisé entraîné)
├── requirements.txt          (Liste des dépendances Python)
├── README.md                 (Ce fichier)
└── generate_mock_data.py     (Script pour générer des données simulées)
```

## Prérequis

*   Python 3.11+
*   `pip` pour la gestion des paquets

## Installation

1.  **Cloner le dépôt (ou dézipper l'archive)**

2.  **Créer et activer un environnement virtuel (recommandé)**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Linux/macOS
    # .venv\Scripts\activate  # Sur Windows PowerShell
    ```

3.  **Installer les dépendances**

    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Acquisition des Données

Le projet est conçu pour fonctionner avec les données originales du Google Drive.

**⚠️ NOTE IMPORTANTE :** Le téléchargement automatique des données originales a échoué en raison de restrictions d'accès ou de limites de `gdown`. Pour que le code fonctionne, vous devez télécharger manuellement le contenu du dossier suivant et le placer dans le répertoire `data/per_point_v2/` :

[https://drive.google.com/drive/folders/1YbrujfBn4kqhrSSqZqG7cpuYs8k4sb2v](https://drive.google.com/drive/folders/1YbrujfBn4kqhrSSqZqG7cpuYs8k4sb2v)

**Alternative (Données Simulées) :**

Si vous ne pouvez pas accéder aux données originales, le projet inclut un script pour générer des données simulées pour le développement et le test :

```bash
python3 generate_mock_data.py
```

### 2. Exécution de la Détection

Le fichier `main.py` contient la logique pour entraîner le modèle supervisé et exécuter les deux méthodes de détection.

```bash
python3 main.py
```

L'exécution de `main.py` effectuera les étapes suivantes :
1.  Charger toutes les données (simulées ou réelles) du dossier `data/per_point_v2`.
2.  Entraîner le modèle supervisé (`RandomForestClassifier`) et le sauvegarder sous `supervised_model.joblib`.
3.  Exécuter les deux méthodes de détection sur un fichier d'échantillon (`ball_data_1.json`).
4.  Afficher les événements détectés et sauvegarder les résultats enrichis dans des fichiers JSON temporaires.

## Méthodes de Détection

### Méthode 1 : Non Supervisée (Basée sur la Physique)

**Fonction :** `unsupervised_hit_bounce_detection(file_path)`

Cette méthode utilise l'analyse des caractéristiques physiques de la trajectoire de la balle :
*   **Calcul des dérivées :** Vitesse (`vx`, `vy`) et Accélération (`ax`, `ay`) sont calculées à partir des positions (x, y).
*   **Détection d'événements :** Un événement (hit ou bounce) est détecté par un pic significatif dans l'accélération totale (`a_total`), indiquant une discontinuité ou un changement de pente.
*   **Différenciation Hit/Bounce :** La distinction est faite de manière heuristique en fonction de la position verticale (`y`) de la balle (un rebond est supposé se produire près du sol, `y > 800` dans le cadre 1080p).

### Méthode 2 : Supervisée (Machine Learning)

**Fonction :** `supervised_hit_bounce_detection(file_path)`

Cette méthode utilise un modèle de Machine Learning entraîné sur les données étiquetées (`action`).
*   **Feature Engineering :** Les caractéristiques incluent les positions, vitesses, accélérations, ainsi que des *lagged features* (valeurs des frames précédentes) pour capturer la dynamique temporelle.
*   **Modèle :** Un `RandomForestClassifier` est utilisé pour classer chaque frame comme `air`, `hit` ou `bounce`.
*   **Entraînement :** Le modèle est entraîné sur l'ensemble des données disponibles et sauvegardé pour une utilisation future.

## Sortie

Les deux fonctions retournent une structure JSON identique à l'entrée, enrichie d'une nouvelle clé : `"pred_action": "hit" / "bounce" / "air"`.

```json
{
  "56100": {
    "x": 894,
    "y": 395,
    "visible": true,
    "action": "air",
    "pred_action": "bounce"
  }
}
```
