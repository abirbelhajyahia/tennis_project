import json
import os
import random
from typing import Dict, Any

def generate_mock_point_data(num_frames: int) -> Dict[str, Dict[str, Any]]:
    """Génère des données de suivi de balle simulées pour un point."""
    data = {}
    actions = ["air"] * (num_frames - 2)
    # Ajouter un hit et un bounce simulés
    actions.insert(random.randint(5, num_frames // 2), "hit")
    actions.insert(random.randint(num_frames // 2 + 1, num_frames - 1), "bounce")
    
    # Simuler une trajectoire simple (chute puis remontée)
    y_start = 300
    y_max = 800
    x_start = 500
    x_end = 1500
    
    for i in range(num_frames):
        frame_number = 56000 + i
        
        # Simuler la position Y (parabole simple)
        t = i / num_frames
        y_pos = int(y_start + (y_max - y_start) * (4 * t * (1 - t)))
        
        # Simuler la position X (mouvement linéaire)
        x_pos = int(x_start + (x_end - x_start) * (i / num_frames))
        
        # Ajouter un peu de bruit
        y_pos += random.randint(-10, 10)
        x_pos += random.randint(-5, 5)
        
        # Assurer que les positions sont dans les limites
        x_pos = max(0, min(1920, x_pos))
        y_pos = max(0, min(1080, y_pos))
        
        # Assigner l'action
        action = actions[i] if i < len(actions) else "air"
        
        data[str(frame_number)] = {
            "x": x_pos,
            "y": y_pos,
            "visible": True,
            "action": action
        }
    return data

def create_mock_dataset(num_files: int = 5, output_dir: str = "data/per_point_v2"):
    """Crée un jeu de données simulé de plusieurs fichiers JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, num_files + 1):
        num_frames = random.randint(50, 150)
        mock_data = generate_mock_point_data(num_frames)
        file_path = os.path.join(output_dir, f"ball_data_{i}.json")
        
        with open(file_path, 'w') as f:
            json.dump(mock_data, f, indent=4)
            
        print(f"Fichier simulé créé: {file_path} avec {num_frames} frames.")

if __name__ == "__main__":
    create_mock_dataset()
