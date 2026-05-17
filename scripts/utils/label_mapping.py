"""
Модуль для работы с объединением меток языков
"""
import pandas as pd

# Маппинг исходных меток на объединённые (ISO 639-1 коды).
# Обновлён под датасет lang_detection_diploma.csv.
LABEL_MAPPING = {
    'hy_arm': 'hy',
    'hy_lat': 'hy',
    'hy_cyr': 'hy',
    'ka_geo': 'ka',
    'ka_lat': 'ka',
    'uz_lat': 'uz',
    'uz_cyr': 'uz',
    'ur_ur': 'ur',
    'ur_lat': 'ur',
    'ne_nep': 'ne',
    'ne_lat': 'ne',
    'sr_cyr': 'sr',
    'sr_lat': 'sr',
    'hi_hi': 'hi',
    'hi_lat': 'ur',
    'kk_lat': 'kk',
    'kk': 'kk',
    'ru_cyr': 'ru',
    'ru_lat': 'ru',
    'am_am': 'am',
    'am_lat': 'am',
    'de': 'other',
    'it': 'other',
    'nl': 'other',
    'pl': 'other',
    'bg': 'other',
    'ja': 'other',
    'vi': 'other',
    'ko': 'other',
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

