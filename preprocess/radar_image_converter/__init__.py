"""
4D Radar Point Cloud to Image Converter

A standalone toolkit for converting 4D radar point clouds (NPY format) 
to BEV (Bird's Eye View) and Range images for deep learning applications.

Author: Extracted from TransLoc4D project
License: Same as TransLoc4D
"""

__version__ = "1.0.0"

from .converters import BEVConverter, RangeImageConverter
from .utils import normalize_image

__all__ = [
    'BEVConverter',
    'RangeImageConverter', 
    'normalize_image'
]
