"""
PlayerTracksDrawer Module
=========================
Dibuja las trayectorias de jugadores rastreados en los fotogramas del video.
Representa cada jugador con una elipse coloreada según su equipo.
"""

from .utils import draw_ellipse


class PlayerTracksDrawer:
    """
    Dibujador de tracks de jugadores coloreados por equipo.
    
    Attributes:
        default_player_team_id (int): Equipo asignado si no se especifica
        team_1_color (tuple): Color BGR para equipo 1
        team_2_color (tuple): Color BGR para equipo 2
    """
    
    def __init__(self, team_1_color=(255, 0, 0), team_2_color=(0, 0, 255)):
        """
        Inicializa el dibujador con colores de equipo.
        
        Args:
            team_1_color (tuple): Color BGR (B, G, R) para el equipo 1. Por defecto rojo (255, 0, 0)
            team_2_color (tuple): Color BGR para el equipo 2. Por defecto azul (0, 0, 255)
        """
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assigment):
        """
        Dibuja los tracks de jugadores en todos los fotogramas.
        
        Itera sobre cada fotograma y dibuja una elipse para cada jugador,
        coloreada según su equipo asignado.
        
        Args:
            video_frames (list): Lista de fotogramas del video
            tracks (list): Tracks de jugadores (estructura: [{track_id: {'bbox': [x1,y1,x2,y2]}, ...}, ...])
            player_assigment (list): Asignación de equipos por fotograma (estructura: [{player_id: team_id, ...}, ...])
            
        Returns:
            list: Lista de fotogramas con tracks dibujados
        """
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            # Crea una copia para evitar modificar el fotograma original
            frame = frame.copy()

            # Obtiene todos los jugadores detectados en este fotograma
            player_dict = tracks[frame_num]

            # Obtiene la asignación de equipos para este fotograma
            player_assigment_for_frame = player_assigment[frame_num]

            # Dibuja a cada jugador detectado
            for track_id, player in player_dict.items():
                # Obtiene el equipo del jugador; usa equipo por defecto si no está asignado
                team_id = player_assigment_for_frame.get(track_id, self.default_player_team_id)
                
                # Selecciona el color según el equipo
                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                # Dibuja una elipse alrededor del jugador con su ID de track
                frame = draw_ellipse(frame, player['bbox'], color, track_id)
            
            output_video_frames.append(frame)
        
        return output_video_frames
    
   