"""
Drawing Utilities Module
========================
Funciones de dibujo para visualizar objetos rastreados en fotogramas de video.
Incluye funciones para dibujar triángulos (pelota) y elipses (jugadores).
"""

import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width

def draw_traingle(frame, bbox, color):
    """
    Dibuja un triángulo en el fotograma para marcar la posición de la pelota.
    
    El triángulo apunta hacia arriba, con la punta en la parte superior del bbox.
    
    Args:
        frame (numpy.ndarray): Fotograma (imagen) donde dibujar
        bbox (list): Bounding box [x1, y1, x2, y2] de la pelota
        color (tuple): Color BGR (B, G, R) del triángulo
        
    Returns:
        numpy.ndarray: Fotograma modificado con el triángulo dibujado
    """
    # Obtiene la coordenada Y superior del bbox
    y = int(bbox[1])
    # Obtiene el centro horizontal del bbox
    x, _ = get_center_of_bbox(bbox)

    # Define los tres puntos del triángulo apuntando hacia arriba
    triangle_points = np.array([
        [x, y],           # Punta superior
        [x - 10, y - 20],  # Esquina inferior izquierda
        [x + 10, y - 20],  # Esquina inferior derecha
    ])
    
    # Dibuja el triángulo relleno con el color especificado
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    # Dibuja el contorno del triángulo en negro
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame


def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Dibuja una elipse alrededor del jugador y opcionalmente su ID de track.
    
    La elipse representa la silueta del jugador y se colorea según su equipo.
    Si se proporciona track_id, dibuja un rectángulo con el número del jugador.
    
    Args:
        frame (numpy.ndarray): Fotograma donde dibujar
        bbox (list): Bounding box [x1, y1, x2, y2] del jugador
        color (tuple): Color BGR de la elipse
        track_id (int, optional): ID del jugador para mostrar en el rectángulo
        
    Returns:
        numpy.ndarray: Fotograma modificado con la elipse dibujada
    """
    # Obtiene la coordenada Y inferior del bbox (parte baja del jugador)
    y2 = int(bbox[3])
    # Obtiene el centro horizontal y vertical del bbox
    x_center, ycenter = get_center_of_bbox(bbox)
    # Calcula el ancho del bbox
    width = get_bbox_width(bbox)

    # Dibuja una elipse en la parte inferior del bbox (simulando los pies del jugador)
    cv2.ellipse(
        frame,
        center=(x_center, y2),                    # Centro en la base del jugador
        axes=(int(width), int(0.35 * width)),    # Ejes de la elipse (ancho y alto)
        angle=0.0,                                # Sin rotación
        startAngle=-45,                           # Ángulo inicial
        endAngle=235,                             # Ángulo final
        color=color,                              # Color del equipo
        thickness=2,                              # Grosor de línea
        lineType=cv2.LINE_4                       # Tipo de línea
    )
    
    # Si se proporciona el ID del jugador, dibuja un rectángulo con el número
    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        # Calcula las coordenadas del rectángulo centrado horizontalmente
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        # Dibuja el rectángulo de fondo para el número del jugador
        cv2.rectangle(frame,
                      (int(x1_rect), int(y1_rect)),
                      (int(x2_rect), int(y2_rect)),
                      color,
                      cv2.FILLED)
        
        # Ajusta la posición X del texto para números de 3 dígitos
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10
        
        # Dibuja el número del jugador en el rectángulo
        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),                             # Texto en negro
            2
        )

    return frame