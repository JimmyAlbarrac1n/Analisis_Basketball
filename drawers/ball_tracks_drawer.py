"""
BallTracksDrawer Module
=======================
Dibuja la posición de la pelota en los fotogramas del video.
Representa la pelota con un triángulo verde apuntando hacia arriba.
"""

from .utils import draw_traingle


class BallTracksDrawer:
    """
    Dibujador del track de la pelota.
    
    Attributes:
        ball_pointer_color (tuple): Color BGR para el triángulo de la pelota
    """
    
    def __init__(self):
        """Inicializa el dibujador de pelota con color verde."""
        # Verde en formato BGR (B=0, G=255, R=0)
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """
        Dibuja la posición de la pelota en todos los fotogramas.
        
        Itera sobre cada fotograma y dibuja un triángulo verde en la posición
        donde se detectó la pelota.
        
        Args:
            video_frames (list): Lista de fotogramas del video
            tracks (list): Tracks de pelota (estructura: [{1: {'bbox': [x1,y1,x2,y2]}, ...}, ...])
                          ID 1 representa la pelota; {} si no se detectó
            
        Returns:
            list: Lista de fotogramas con la posición de la pelota dibujada
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            # Crea una copia para evitar modificar el fotograma original
            frame = frame.copy()
            
            # Obtiene los objetos detectados (pelota) en este fotograma
            ball_dict = tracks[frame_num]

            # Dibuja la pelota si existe en este fotograma
            for _, ball in ball_dict.items():
                # Si no hay bbox válida, continúa con el siguiente
                if ball["bbox"] is None:
                    continue
                
                # Dibuja un triángulo verde en la posición de la pelota
                frame = draw_traingle(frame, ball["bbox"], self.ball_pointer_color)

            output_video_frames.append(frame)
        
        return output_video_frames