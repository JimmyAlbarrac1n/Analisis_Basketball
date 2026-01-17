"""
PlayerTracker Module
====================
Detecta y rastrea jugadores en videos de baloncesto utilizando YOLO y ByteTrack.
Proporciona funcionalidad para detectar fotogramas, asignar IDs únicos a jugadores
y almacenar/recuperar datos de seguimiento (stubs).
"""

from ultralytics import YOLO
import supervision as sv
import sys
sys.path.append('../')
from utils import read_stub, save_stub

class PlayerTracker:
    """
    Rastreador de jugadores usando YOLO para detección y ByteTrack para seguimiento.
    
    Attributes:
        model (YOLO): Modelo YOLO entrenado para detectar jugadores
        tracker (sv.ByteTrack): Sistema de seguimiento que asigna IDs únicos a jugadores
    """
    
    def __init__(self, model_path):
        """
        Inicializa el rastreador de jugadores.
        
        Args:
            model_path (str): Ruta al modelo YOLO entrenado (ej: 'models/player_detector.pt')
        """
        # Carga el modelo YOLO preentrenado
        self.model = YOLO(model_path) 
        # Inicializa el tracker ByteTrack para asignar IDs consistentes a jugadores
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """
        Ejecuta inferencia YOLO en lotes de fotogramas para detectar jugadores.
        
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
        Obtiene el rastreo de jugadores en todos los fotogramas.
        Intenta leer desde cache (stub) primero para evitar reprocessamiento.
        
        Args:
            frames (list): Lista de fotogramas del video
            read_from_stub (bool): Si es True, intenta leer datos cacheados
            stub_path (str): Ruta al archivo de cache
            
        Returns:
            list: Lista de diccionarios con tracks. Estructura:
                  [{track_id_1: {'bbox': [x1, y1, x2, y2]}, track_id_2: {...}}, ...]
                  Cada índice representa un fotograma
        """
        # Intenta cargar datos previamente procesados si existen
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        # Si no hay cache válido, procede con detección y tracking
        detections = self.detect_frames(frames)

        tracks = []

        for frame_num, detection in enumerate(detections):
            # Crea mapeo inverso de nombres de clases (id -> nombre)
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convierte las detecciones YOLO al formato de supervision
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Actualiza el tracker con las nuevas detecciones
            # ByteTrack asigna IDs consistentes a los jugadores entre fotogramas
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Inicializa el diccionario de tracks para este fotograma
            tracks.append({})

            # Procesa cada detección con su ID de seguimiento
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # Solo guarda detecciones que corresponden a jugadores
                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox": bbox}
        
        # Guarda los tracks en cache para futuras ejecuciones
        save_stub(stub_path, tracks)
        return tracks