"""
Video Utilities Module
=====================
Funciones auxiliares para lectura y escritura de videos.
Gestiona la carga de videos en memoria como listas de fotogramas
y la escritura de fotogramas procesados en archivos de video.
"""

import cv2
import os

def read_video(video_path):
    """
    Lee un video y retorna una lista con todos sus fotogramas.
    
    Carga completamente el video en memoria como una lista de arrays numpy.
    Cada fotograma está en formato BGR (OpenCV).
    
    Args:
        video_path (str): Ruta del archivo de video (ej: 'input_videos/video.mp4')
        
    Returns:
        list: Lista de fotogramas (numpy.ndarray) en formato BGR
        
    Note:
        - Requiere suficiente memoria RAM para almacenar todo el video
        - Para videos largos, puede ser intensivo en memoria
    """
    # Abre el archivo de video
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Lee fotogramas secuencialmente hasta el final del video
    while True:
        ret, frame = cap.read()
        if not ret:  # Si no se puede leer más fotogramas, sale del bucle
            break
        frames.append(frame)
    
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """
    Guarda una lista de fotogramas como un archivo de video.
    
    Crea el directorio de salida si no existe y escribe los fotogramas
    en un archivo de video con codec XVID a 20 fps.
    
    Args:
        output_video_frames (list): Lista de fotogramas (numpy.ndarray) en formato BGR
        output_video_path (str): Ruta donde guardar el video (ej: 'output_videos/result.avi')
        
    Returns:
        None
        
    Note:
        - Usa codec XVID (.avi) por compatibilidad
        - Frame rate fijo en 20 fps
        - Resuelve automáticamente el tamaño del fotograma del primer frame
    """
    # Crea el directorio de salida si no existe
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.mkdir(os.path.dirname(output_video_path))

    # Define el codec de video (XVID es compatible y comprime bien)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Obtiene el ancho y alto del primer fotograma
    frame_height, frame_width = output_video_frames[0].shape[0], output_video_frames[0].shape[1]
    
    # Crea el objeto VideoWriter
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    
    # Escribe cada fotograma en el video
    for frame in output_video_frames:
        out.write(frame)
    
    # Cierra el archivo de video (importante para finalizar la escritura)
    out.release()