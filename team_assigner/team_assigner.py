"""
TeamAssigner Module
===================
Asigna jugadores a equipos basándose en el color de su uniforme usando visión por computadora.
Utiliza el modelo CLIP preentrenado de Fashion para clasificar tipos de uniformes.
"""

from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
import sys 
sys.path.append('../')
from utils import read_stub, save_stub

from utils.stubs_utils import read_stub


class TeamAssigner:
    """
    Asignador de equipos a jugadores basado en análisis de uniformes.
    
    Utiliza el modelo CLIP de Fashion para interpretar la ropa del jugador
    y clasificarla como perteneciente a uno de dos equipos.
    
    Attributes:
        player_team_dict (dict): Cache que mapea IDs de jugador a números de equipo
        team_1_class_name (str): Descripción del uniforme del equipo 1
        team_2_class_name (str): Descripción del uniforme del equipo 2
        model (CLIPModel): Modelo de visión-lenguaje para clasificar ropa
        processor (CLIPProcessor): Procesador para preparar imágenes y texto
    """
    
    def __init__(self,
                 team_1_class_name="white shirt",
                 team_2_class_name="dark blue shirt"):
        """
        Inicializa el asignador de equipos.
        
        Args:
            team_1_class_name (str): Descripción textual del uniforme del equipo 1
            team_2_class_name (str): Descripción textual del uniforme del equipo 2
        """
        self.team_colors = {}
        # Cache para evitar reclasificar el mismo jugador múltiples veces
        self.player_team_dict = {}        
    
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        """
        Carga el modelo CLIP de Fashion y su procesador.
        
        Este es un modelo preentrenado que puede entender descripciones de ropa
        (ej: "white shirt", "dark blue shirt") y compararlas con imágenes.
        """
        # Modelo CLIP específicamente entrenado en Fashion
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        # Procesador para preparar imágenes y textos para el modelo
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    

    def get_player_color(self, frame, bbox):
        """
        Clasifica el color/tipo de uniforme del jugador usando el modelo CLIP.
        
        Extrae la región del jugador del fotograma y usa CLIP para clasificar
        si se parece más al "white shirt" o "dark blue shirt".
        
        Args:
            frame (numpy.ndarray): Fotograma del video (BGR)
            bbox (list): Bounding box [x1, y1, x2, y2] del jugador
            
        Returns:
            str: Nombre de la clase ("white shirt" o "dark blue shirt")
        """
        # Extrae la región de interés (ROI) del jugador del fotograma
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Convierte de BGR (OpenCV) a RGB (PIL/CLIP)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        image = pil_image

        # Define las dos clases a clasificar
        classes = [self.team_1_class_name, self.team_2_class_name]

        # Prepara la imagen y el texto para el modelo CLIP
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True)

        # Realiza la predicción con CLIP
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        # Calcula probabilidades usando softmax
        probs = logits_per_image.softmax(dim=1)

        # Retorna la clase con mayor probabilidad
        class_name = classes[probs.argmax(dim=1)[0]]

        return class_name

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determina el equipo de un jugador específico.
        
        Verifica primero si el jugador ya fue clasificado (cache).
        Si no, clasifica su uniforme usando get_player_color().
        
        Args:
            frame (numpy.ndarray): Fotograma del video
            player_bbox (list): Bounding box del jugador
            player_id (int): ID único del jugador
            
        Returns:
            int: Número de equipo (1 o 2)
        """
        # Verifica si el jugador ya fue clasificado previamente
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Clasifica el uniforme del jugador
        player_color = self.get_player_color(frame, player_bbox)

        # Asigna número de equipo basado en la clasificación del uniforme
        team_id = 2  # Por defecto, equipo 2
        if player_color == self.team_1_class_name:
            team_id = 1

        # Guarda el resultado en cache para futuras consultas
        self.player_team_dict[player_id] = team_id
        return team_id
    
    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        """
        Asigna equipos a todos los jugadores en todos los fotogramas.
        
        Procesa cada fotograma y cada jugador, asignando un número de equipo
        basado en su uniforme. Intenta usar datos en cache si están disponibles.
        
        Args:
            video_frames (list): Lista de fotogramas del video
            player_tracks (list): Tracks de jugadores de PlayerTracker.get_object_tracks()
            read_from_stub (bool): Si es True, intenta leer datos cacheados
            stub_path (str): Ruta al archivo de cache
            
        Returns:
            list: Lista de diccionarios. Estructura:
                  [{player_id: team_id, ...}, {}, ...]
                  Cada índice es un fotograma con mapeo de jugador->equipo
        """
        # Intenta cargar asignaciones previamente calculadas
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        # Si no hay cache válido, procede con clasificación
        self.load_model()

        player_assignment = []
        for frame_num, player_track in enumerate(player_tracks):        
            # Inicializa diccionario vacío para este fotograma
            player_assignment.append({})
            
            # Limpia el cache de clasificación cada 50 fotogramas
            # Esto permite que el modelo se reajuste a cambios de iluminación o perspectiva
            if frame_num % 50 == 0:
                self.player_team_dict = {}

            # Clasifica cada jugador en este fotograma
            for player_id, track in player_track.items():
                # Obtiene el equipo del jugador
                team = self.get_player_team(video_frames[frame_num],   
                                            track['bbox'],
                                            player_id)
                player_assignment[frame_num][player_id] = team
        
        # Guarda las asignaciones en cache para futuras ejecuciones
        save_stub(stub_path, player_assignment)

        return player_assignment