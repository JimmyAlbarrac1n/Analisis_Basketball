"""
BallTracker Module
==================
Detecta y rastrea la pelota de baloncesto en videos utilizando YOLO.
Incluye funcionalidad para remover detecciones erróneas e interpolar posiciones.
"""

from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import read_stub, save_stub


class BallTracker:
    """
    Rastreador de pelota usando YOLO para detección.
    
    A diferencia de PlayerTracker, no usa un tracker persistente porque
    la pelota no necesita IDs únicos: se selecciona la detección con mayor confianza.
    
    Attributes:
        model (YOLO): Modelo YOLO entrenado para detectar pelotas
    """
    
    def __init__(self, model_path):
        """
        Inicializa el rastreador de pelota.
        
        Args:
            model_path (str): Ruta al modelo YOLO entrenado (ej: 'models/ball_detector.pt')
        """
        # Carga el modelo YOLO preentrenado para detección de pelotas
        self.model = YOLO(model_path) 

    def detect_frames(self, frames):
        """
        Ejecuta inferencia YOLO en lotes de fotogramas para detectar pelotas.
        
        Args:
            frames (list): Lista de fotogramas (imágenes numpy) del video
            
        Returns:
            list: Lista de objetos Detection de YOLO con detecciones por fotograma
        """
        # Procesa fotogramas en lotes para optimizar memoria y velocidad
        batch_size=20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            # Predice sobre un lote de fotogramas con confianza mínima del 50%
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.5)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Obtiene la detección de pelota en todos los fotogramas.
        Selecciona la detección con mayor confianza por fotograma.
        
        Args:
            frames (list): Lista de fotogramas del video
            read_from_stub (bool): Si es True, intenta leer datos cacheados
            stub_path (str): Ruta al archivo de cache
            
        Returns:
            list: Lista de diccionarios con tracks de pelota. Estructura:
                  [{1: {'bbox': [x1, y1, x2, y2]}}, {1: {...}}, ...]
                  ID 1 representa la pelota; None si no se detectó en ese fotograma
        """
        # Intenta cargar datos previamente procesados si existen
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        # Si no hay cache válido, procede con detección
        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convierte las detecciones YOLO al formato de supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Inicializa el diccionario de tracks para este fotograma
            tracks.append({})
            chosen_bbox = None
            max_confidence = 0
            
            # Busca la detección de pelota con mayor confianza
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]
                
                # Solo considera detecciones que correspondan a pelotas
                if cls_id == cls_names_inv['Ball']:
                    # Mantiene la detección con mayor confianza
                    if max_confidence < confidence:
                        chosen_bbox = bbox
                        max_confidence = confidence

            # Guarda la pelota detectada (si existe)
            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        # Guarda los tracks en cache para futuras ejecuciones
        save_stub(stub_path, tracks)
        
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Elimina detecciones de pelota que son imposibles debido a movimiento erróneo.
        
        Si la pelota se mueve más de una distancia máxima permitida entre fotogramas,
        marca el fotograma como sin detección válida.
        
        Args:
            ball_positions (list): Tracks de pelota de get_object_tracks()
            
        Returns:
            list: Ball_positions filtrado con detecciones erroneas eliminadas
        """
        # Distancia máxima permitida en pixels que la pelota puede moverse entre fotogramas consecutivos
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            # Obtiene el bbox de la pelota en el fotograma actual
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            # Si no hay detección en este fotograma, continúa
            if len(current_box) == 0:
                continue

            # Si es la primera detección válida, guárdala como referencia
            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            # Obtiene el bbox del último fotograma con detección válida
            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            # Calcula el número de fotogramas entre detecciones válidas
            frame_gap = i - last_good_frame_index
            # Ajusta la distancia máxima permitida según el número de fotogramas (permite más movimiento en gaps grandes)
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            # Calcula la distancia euclidiana entre centros de las dos detecciones
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                # Si la distancia es demasiada grande, descarta esta detección como errónea
                ball_positions[i] = {}
            else:
                # Si es válida, actualiza la referencia del último fotograma bueno
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpola posiciones faltantes de la pelota usando interpolación lineal.
        
        Si la pelota no se detecta en algunos fotogramas intermedios,
        esta función predice su posición basándose en las detecciones válidas.
        
        Args:
            ball_positions (list): Tracks de pelota (con posibles huecos)
            
        Returns:
            list: Ball_positions con todas las posiciones interpoladas
        """
        # Extrae solo los valores de bbox (x1, y1, x2, y2) de cada fotograma
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        
        # Convierte a DataFrame de pandas para usar funciones de interpolación
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolación lineal: rellena valores faltantes con interpolación
        df_ball_positions = df_ball_positions.interpolate()
        # Rellena los valores al principio/final con el método backward fill (último valor válido)
        df_ball_positions = df_ball_positions.bfill()

        # Convierte el DataFrame de vuelta al formato original
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions