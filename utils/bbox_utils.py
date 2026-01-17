"""
Bounding Box Utilities Module
=============================
Funciones auxiliares para operaciones con bounding boxes.
Proporciona utilidades para calcular centros y dimensiones de cajas delimitadoras.
"""

def get_center_of_bbox(bbox):
    """
    Calcula el centro de un bounding box.
    
    Args:
        bbox (list): Bounding box en formato [x1, y1, x2, y2]
                     donde (x1, y1) es la esquina superior izquierda
                     y (x2, y2) es la esquina inferior derecha
        
    Returns:
        tuple: Coordenadas (x, y) del centro del bbox (valores enteros)
        
    Example:
        >>> center = get_center_of_bbox([10, 20, 110, 120])
        >>> print(center)
        (60, 70)
    """
    x1, y1, x2, y2 = bbox
    # Calcula el promedio de las coordenadas X e Y
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """
    Calcula el ancho de un bounding box.
    
    Args:
        bbox (list): Bounding box en formato [x1, y1, x2, y2]
        
    Returns:
        int: Ancho del bbox (diferencia entre x2 y x1)
        
    Example:
        >>> width = get_bbox_width([10, 20, 110, 120])
        >>> print(width)
        100
    """
    # El ancho es la diferencia entre la coordenada X derecha e izquierda
    return bbox[2] - bbox[0]