"""
Модуль для работы с объединением меток языков
"""
import pandas as pd

# Маппинг исходных меток на объединенные
LABEL_MAPPING = {
    'uz_lat': 'uz',
    'uz_kir': 'uz',
    'ur_lat': 'ur',
    'ur_arab': 'ur',
    'ne': 'ne',
    'ne_lat': 'ne',
    'sr_lat': 'sr',
    'sr_cyr': 'sr',
}

def merge_label(label):
    """
    Объединяет метку языка в итоговую метку
    
    Args:
        label: исходная метка (например, 'uz_lat', 'uz_kir')
    
    Returns:
        объединенная метка (например, 'uz') или исходная, если нет маппинга
    """
    if pd.isna(label):
        return label
    
    label_str = str(label).strip()
    return LABEL_MAPPING.get(label_str, label_str)

def merge_labels_in_series(series):
    """
    Объединяет метки в pandas Series
    
    Args:
        series: pandas Series с метками
    
    Returns:
        pandas Series с объединенными метками
    """
    import pandas as pd
    return series.apply(merge_label)

def get_original_labels():
    """
    Возвращает список всех исходных меток, которые нужно объединить
    
    Returns:
        список исходных меток
    """
    return list(LABEL_MAPPING.keys())

def get_merged_labels():
    """
    Возвращает список всех объединенных меток
    
    Returns:
        список объединенных меток
    """
    return list(set(LABEL_MAPPING.values()))

