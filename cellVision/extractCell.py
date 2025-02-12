import os
from cellSAM import segment_cellular_image, get_model
import numpy as np
import torch
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class extractCell:
    def __init__(self, input_path, cell_img_save_path, minimum_contour_area=500):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img = np.array(Image.open(input_path))
        self.minimum_contour_area = minimum_contour_area
        self.cell_img_save_path = cell_img_save_path
        self.img_name = input_path.split("/")[-1].split(".")[0]
        if not os.path.exists(self.cell_img_save_path):
            os.makedirs(self.cell_img_save_path)

    @staticmethod
    def get_segmentation_result(img, device):
        mask, embedding, bounding_boxes = segment_cellular_image(img, device=str(device), normalize=True, bbox_threshold=0.2)
        mask = cv2.copyMakeBorder(mask, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return mask

    @staticmethod
    def find_edges_for_masks(mask):
        dilated_mask = binary_dilation(mask)
        edges = dilated_mask ^ mask  # XOR operation to find edges
        return edges
    
    @staticmethod
    def save_cells(img, image_name, edges, out_folder, minimum_contour_area):
        if len(img.shape) == 2 or img.shape[-1] != 3:
            img = np.stack([img] * 3, axis=-1)

        img = cv2.copyMakeBorder(img, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        binary_mask = np.where(np.isclose(1.0, edges), 1, 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for object_id, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)

            if contour_area > minimum_contour_area:
                blank_mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.fillPoly(blank_mask, [contour], (255,255,255))
                blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)

                result = cv2.bitwise_and(img,img,mask=blank_mask)

                output_path = f'{out_folder}/{image_name}_{object_id + 1}.png'
                Image.fromarray(result).save(output_path)
    
    def extract_and_save_cells(self):
        mask = self.get_segmentation_result(self.img, self.device)
        edges = self.find_edges_for_masks(mask)
        self.save_cells(self.img, self.img_name, edges, self.cell_img_save_path, self.minimum_contour_area)
