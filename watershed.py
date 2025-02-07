import os
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.measure import regionprops

class Watershed:
    def __init__(self, cell_image_folder: str, large_mask_threshold = 220, small_mask_threshold = 160, low_color_threshold = [30, 110, 109], high_color_threshold = [101, 255, 255]):
        self.cell_input_folder = cell_image_folder
        self.base_out_dir = os.path.dirname(self.cell_input_folder)
        self.original_img_name = self.cell_input_folder.split("/")[-1]
        self.out_folder_name = f"watershed_markers_{self.original_img_name}"
        if not os.path.exists(os.path.join(self.base_out_dir, self.out_folder_name)):
            os.makedirs(os.path.join(self.base_out_dir, self.out_folder_name))
        
        self.marker_saving_path = os.path.join(self.base_out_dir, self.out_folder_name)
        self.large_mask_threshold = large_mask_threshold
        self.small_mask_threshold = small_mask_threshold
        self.low_color_threshold = low_color_threshold
        self.high_color_threshold = high_color_threshold

    @staticmethod
    def extract_neon_mask(image, low_color_threshold = [30, 110, 109], high_color_threshold = [101, 255, 255]):
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_green = np.array(low_color_threshold)
        high_green = np.array(high_color_threshold)

        neon_mask = cv2.inRange(img_hsv, low_green, high_green)

        result = cv2.bitwise_and(image, image, mask=neon_mask)
        return result

    @staticmethod
    def threshold_image(image, low_threshold):
        gray_img = image[:, :, 1]
        _, bright_mask = cv2.threshold(gray_img, low_threshold, 255, cv2.THRESH_BINARY)  
        return bright_mask

    @staticmethod
    def watershed_for_large_blobs(binary):
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        dist_normalized = np.uint8(dist_normalized)
        
        # Find sure background
        kernel = np.ones((3,3), np.uint8)
        sure_bg = cv2.dilate(binary, kernel, iterations=5)
        
        # sure foreground (peaks in distance transform)
        # Threshold distance transform to get peaks
        _, sure_fg = cv2.threshold(dist_normalized, 0.3*dist_normalized.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
    
        large_markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)
        unique_elements, counts = np.unique(large_markers, return_counts=True)
        max_index = np.argmax(counts)
        large_markers[large_markers == unique_elements[max_index]] = 0
        large_markers[large_markers == -1] = 0
        
        return large_markers, dist_transform
    
    @staticmethod
    def watershed_for_small_blobs(binary):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        small_blobs_mask = np.zeros_like(binary)
        
        min_blob_size = 2 
        max_small_blob_size = 200
        
        # Separate small and large blobs
        for i in range(1, num_labels):  
            area = stats[i, cv2.CC_STAT_AREA]
            if min_blob_size <= area <= max_small_blob_size:
                blob_mask = (labels == i).astype(np.uint8) * 255
                small_blobs_mask = cv2.bitwise_or(small_blobs_mask, blob_mask)
        
        if np.any(small_blobs_mask):
            dist_transform = cv2.distanceTransform(small_blobs_mask, cv2.DIST_L2, 3)

            dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
            _, sure_fg = cv2.threshold(dist_normalized, 0.2*dist_normalized.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            coordinates = peak_local_max(dist_transform, min_distance=2)
            markers = np.zeros(binary.shape, dtype=np.int32)
            for i, (x, y) in enumerate(coordinates, start=1):
                markers[x, y] = i

            markers = cv2.watershed(cv2.cvtColor(small_blobs_mask, cv2.COLOR_GRAY2BGR), markers)
            
            small_result = np.zeros_like(binary)
            small_result[markers > 0] = 255
        else:
            small_result = np.zeros_like(binary)
            markers = np.zeros_like(binary)
        
        final_result = small_result
        
        return final_result, markers, dist_transform if np.any(small_blobs_mask) else np.zeros_like(binary)

    @staticmethod
    def small_blobs_unique_labels(large_markers, small_markers):
        add_label_to_small_marker = np.max(large_markers)
        small_markers[small_markers > 0] += add_label_to_small_marker
        unique_elements, counts = np.unique(small_markers, return_counts=True)
        max_index = np.argmax(counts)
        small_markers[small_markers == unique_elements[max_index]] = 0
        small_markers[small_markers == -1] = 0
        return small_markers
        
    @staticmethod
    def merge_small_and_large_markers(image1, image2):
        merged_image = np.zeros_like(image1, dtype=np.int32)
        props1 = regionprops(image1)
        props2 = regionprops(image2)

        regions1 = {prop.label: prop for prop in props1}
        regions2 = {prop.label: prop for prop in props2}

        used_regions2 = set()

        # Process each region in image1
        for label1, prop1 in regions1.items():
            mask1 = image1 == label1
            
            # Check overlap with regions in image2
            overlap_labels = np.unique(image2[mask1])
            overlap_labels = overlap_labels[overlap_labels > 0] 
            
            if len(overlap_labels) == 0:
                merged_image[mask1] = label1
            else:
                max_area = prop1.area
                best_label = label1
                for label2 in overlap_labels:
                    if label2 in regions2:
                        area2 = regions2[label2].area
                        if area2 > max_area:
                            max_area = area2
                            best_label = label2
                        used_regions2.add(label2)
                
                merged_image[mask1] = best_label

        # Add remaining regions from image2 that were not used
        for label2, prop2 in regions2.items():
            if label2 not in used_regions2:
                merged_image[image2 == label2] = label2
                
        # for i in range(len(image2)):
        #     for j in range(len(image2[0])):
        #         if image2[i][j] != 0:
        #             if merged_image[i][j] == 0:
        #                 merged_image[i][j] = image2[i][j]
        
        return merged_image

    def apply_watershed_process_on_folder(self):
        total = len(os.listdir(self.cell_input_folder))
        props = []
        saved_region_names = []
        for c, img in enumerate(os.listdir(self.cell_input_folder)):
            image = cv2.imread(os.path.join(self.cell_input_folder, img))
            if image is None:
                print(f"Skipping {img}: Unable to read the image.")
                continue
            neon_mask = self.extract_neon_mask(image, self.low_color_threshold, self.high_color_threshold)
            # plt.imsave(os.path.join('/home/pamin/Meet_proj/cellSAM/neon_threshold', f"{img.split('.')[0]}_neon_mask.png"), neon_mask)

            #process for large mask
            thresholded_img = self.threshold_image(neon_mask, self.large_mask_threshold)
            large_markers, _ = self.watershed_for_large_blobs(thresholded_img)
            print(f"INFO: large blob markers extracted")

            #process for small mask
            thresholded_img = self.threshold_image(neon_mask, self.small_mask_threshold)
            _, small_markers, _  = self.watershed_for_small_blobs(thresholded_img)
            print(f"INFO: small blob markers extracted")

            #making both large and small markers labels unique
            small_markers = self.small_blobs_unique_labels(large_markers, small_markers)

            #combining both the markers for final saving and returing properties for further processing
            combined_markers = self.merge_small_and_large_markers(large_markers, small_markers)
            try:
                properties_of_markers = regionprops(combined_markers)
            except:
                print("couldn't extract properties of segmented markers check if they are extracted properly")
                properties_of_markers = None

            plt.imsave(os.path.join(self.marker_saving_path, img), combined_markers, cmap = "nipy_spectral")
            saved_region_names.append(img)
            print(f"INFO: {c+1}/{total} images processed")
            props.append(properties_of_markers)
        return props, saved_region_names

    @staticmethod
    def apply_watershed_on_image(img : np.ndarray , large_mask_threshold = 220, small_mask_threshold = 160):
        neon_mask = Watershed.extract_neon_mask(img)

        #process for larger markers
        thresholded_img = Watershed.threshold_image(neon_mask, large_mask_threshold)
        large_markers = Watershed.watershed_for_large_blobs(thresholded_img)

        #process for small markers
        thresholded_img = Watershed.threshold_image(neon_mask, small_mask_threshold)
        small_markers = Watershed.watershed_for_small_blobs(thresholded_img)

        #making both large and small markers labels unique
        small_markers = Watershed.small_blobs_unique_labels(large_markers, small_markers)
        
        #combining both the markers and returning it
        combined_markers = Watershed.merge_small_and_large_markers(large_markers, small_markers)        
        return combined_markers
