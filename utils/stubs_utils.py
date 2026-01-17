"""
Stubs Utilities Module
=====================
Funciones para cachear (guardar y cargar) datos procesados usando pickle.
Permite reutilizar resultados de procesamiento costoso (tracks, clasificaciones)
sin necesidad de volver a procesarlos.
"""

import os
import pickle

def save_stub(stub_path, object):
    """
    Guarda un objeto en un archivo pickle (cache).
    
    Útil para guardar resultados de procesamiento costoso como:
    - Detecciones y tracks de objetos
    - Asignaciones de equipos
    - Interpolaciones de posición
    
    Args:
        stub_path (str): Ruta donde guardar el archivo pickle
                        (ej: 'stubs/player_track_stubs.pkl')
        object (any): Objeto Python para serializar y guardar
        
    Returns:
        None
        
    Note:
        - Crea automáticamente directorios si no existen
        - Sobrescribe archivos existentes
        - El objeto debe ser serializable por pickle
    """
    # Crea el directorio si no existe
    if not os.path.exists(os.path.dirname(stub_path)):
        os.makedirs(os.path.dirname(stub_path))

    # Guarda el objeto solo si se proporciona una ruta válida
    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)

def read_stub(read_from_stub, stub_path):
    """
    Carga un objeto desde un archivo pickle (cache).
    
    Intenta cargar datos previamente procesados para evitar
    procesamiento duplicado.
    
    Args:
        read_from_stub (bool): Si es True, intenta leer del cache
        stub_path (str): Ruta del archivo pickle a cargar
        
    Returns:
        object or None: El objeto cacheado si existe, None si no se puede cargar
        
    Note:
        - Retorna None si read_from_stub es False
        - Retorna None si el archivo no existe
        - Los datos están en el formato que se guardaron originalmente
    """
    # Solo intenta leer si está habilitado y la ruta es válida
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            object = pickle.load(f)
            return object
    
    # Retorna None si no se pudo cargar o no estaba habilitado
    return None
    