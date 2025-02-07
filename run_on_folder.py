import os
from pathlib import Path
from watershed import Watershed
from extractCell import extractCell
from processProps import ProcessProps
import argparse
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--folder-input-path", type=str, help="path to input image")
parser.add_argument("--main-image-folder", type=str, help="parent folder where all the images are stored")

'''
    CONSTANTS: change it according to image conditions
'''

#for proper black and white bg with neon lipids
# LOW_THRESHOLD = [30, 111, 30]
# HIGH_THRESHOLD = [100, 255, 255]
# SMALL_MASK_THRESHOLD = 30

#for greenish bg with neon lipids
LOW_THRESHOLD = [30, 111, 57]
HIGH_THRESHOLD = [101, 255, 255]
SMALL_MASK_THRESHOLD = 40


args = parser.parse_args()
parent_folder = "output"

if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)

folder_dir = args.folder_input_path.split(args.main_image_folder + "/")[0]
image_dir = args.folder_input_path.split(args.main_image_folder + "/")[1]

output_dir = folder_dir + parent_folder + "/" + image_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


#using cellSAM model to extract the cells from the image
for f in os.listdir(args.folder_input_path):
    if f.endswith(".png") or f.endswith(".jpg"):
        image_path = os.path.join(args.folder_input_path, f)
        cell_save_dir_name = f.split(".")[0] + "_cells"
        extract_cells = extractCell(image_path, os.path.join(output_dir, cell_save_dir_name))
        extract_cells.extract_and_save_cells()
        print("cell extraction complete!")

        #using watershed technique to extract lipids from the extracted cells
        watershed = Watershed(os.path.join(output_dir, cell_save_dir_name), small_mask_threshold=SMALL_MASK_THRESHOLD, 
                            low_color_threshold=LOW_THRESHOLD, 
                            high_color_threshold=HIGH_THRESHOLD)

        props, region_names = watershed.apply_watershed_process_on_folder()

        names = []
        cell_counts = []
        areas = []
        centroids = []
        avg_areas = []
        avg_euclidean_distances = []

        for prop, region_name in zip(props, region_names):
            process_props = ProcessProps(prop, region_name)
            extracted_props = process_props.save_aggregated_properties_as_json()
            p = json.loads(extracted_props)
            names.append(p["Image Name"])
            cell_counts.append(p["Cell Count"])
            areas.append(p["area"])
            centroids.append(p["centroids"])
            avg_areas.append(p["Average Area"])
            avg_euclidean_distances.append(p["Average Euclidean Distance"])
            
        

        df = pd.DataFrame({
            "Image Name": names,
            "Cell Count": cell_counts,
            "Area": areas,
            "Centroids": centroids,
            "Average Area": avg_areas,
            "Average Euclidean Distance": avg_euclidean_distances
        })
        print(df)
        prop_file_name = 'properties_' + f.split(".")[0] + '.xlsx'
        with pd.ExcelWriter(os.path.join(output_dir, prop_file_name)) as writer:
            df.to_excel(writer, index=False)
        
        print("properties saved in excel file")