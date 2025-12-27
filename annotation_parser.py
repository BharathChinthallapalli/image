import numpy as np
import cv2
import xml.etree.ElementTree as etree
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import CONFIG
from WSI_load import WSIHandler

class AnnotationParser:
    """
    Parses Camelyon-style XML annotations and generates masks.
    """

    def __init__(self, xml_path):
        """
        Initialize the parser and load annotations.
        
        Args:
            xml_path: Path to the XML annotation file.
        """
        self.xml_path = xml_path
        self.polygons = self._parse_xml()

    def _parse_xml(self):
        """Parse XML to extract polygon coordinates."""
        try:
            tree = etree.parse(self.xml_path)
            root = tree.getroot()
            annotations = root.findall(".//Annotation")

            polygons = []
            for ann in annotations:
                coords = []
                for c in ann.findall(".//Coordinate"):
                    x = float(c.attrib["X"])
                    y = float(c.attrib["Y"])
                    coords.append((x, y))
                
                if len(coords) >= 3:
                    polygons.append(Polygon(coords))
            
            print(f"[AnnotationParser] Parsed {len(polygons)} polygons from {self.xml_path}")
            return polygons
        except Exception as e:
            print(f"[AnnotationParser] Error parsing XML: {e}")
            return []

    def create_mask(self, shape, downsample_factor, tumor_only=True):
        """
        Create a binary mask from the annotations.

        Args:
            shape: (height, width) of the output mask.
            downsample_factor: Factor to scale down the coordinates (WSI_level0 / mask_resolution).
            tumor_only: If True, assumes all annotations are tumor (standard for Camelyon).
        
        Returns:
            mask: Binary mask (numpy array) of shape `shape`.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if not self.polygons:
            return mask

        # Scale polygons to the target resolution
        for poly in self.polygons:
            # Exterior coordinates
            ext_coords = np.array(poly.exterior.coords) / downsample_factor
            ext_pts = ext_coords.astype(np.int32).reshape((-1, 1, 2))
            
            # Draw the polygon
            cv2.fillPoly(mask, [ext_pts], 1)

        return mask
