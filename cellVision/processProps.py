from skimage.measure import regionprops
import numpy as np
import json

class ProcessProps:
    def __init__(self, props, image_name):
        self.props = props
        self.image_name = image_name
        self.cell_count = len(self.props)
        self.cell_areas = [prop.area for prop in self.props]
        self.cell_centroids = [prop.centroid for prop in self.props]
    
    def calculate_avg_distance(self):
        if self.cell_count < 2:
            return 0
        
        euclidean_distance = 0
        for i in range(self.cell_count):
            for j in range(i+1, self.cell_count):
                euclidean_distance += np.linalg.norm(np.array(self.cell_centroids[i]) - np.array(self.cell_centroids[j]))
        return euclidean_distance / (self.cell_count * (self.cell_count - 1) / 2)
    
    def calculate_avg_area(self):
        if self.cell_count == 0:
            return 0
        return sum(self.cell_areas) / self.cell_count
    
    def save_aggregated_properties_as_json(self):
        if self.cell_count == 0:
            return json.dumps({"Image Name": self.image_name, "Cell Count": 0, "area": None, "centroids": None, "Average Area": 0, "Average Euclidean Distance": 0})
        
        avg_euclidian_distance = self.calculate_avg_distance()
        avg_area = self.calculate_avg_area()
        data = {
            "Image Name": self.image_name,
            "Cell Count": self.cell_count,
            "area": ' '.join(map(str, self.cell_areas)),
            "centroids": ' '.join(map(str, self.cell_centroids)),
            "Average Area": avg_area,
            "Average Euclidean Distance": avg_euclidian_distance
        }

        return json.dumps(data)