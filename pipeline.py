from watershed import Watershed
from extractCell import extractCell
from processProps import ProcessProps
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img-input-path", type=str, help="path to input image")
parser.add_argument("--cell-save-path", type=str, help="path to save extracted cells using cellSAM model")

args = parser.parse_args()

#using cellSAM model to extract the cells from the image
extract_cells = extractCell(args.img_input_path, args.cell_save_path)
extract_cells.extract_and_save_cells()
print("cell extraction complete!")

#using watershed technique to extract lipids from the extracted cells
watershed = Watershed(args.cell_save_path, small_mask_threshold=50, low_color_threshold=[30, 111, 30], high_color_threshold=[100, 255, 255])
props, region_names = watershed.apply_watershed_process_on_folder()

for prop, region_name in zip(props, region_names):
    process_props = ProcessProps(prop, region_name)
    print(process_props.save_aggregated_properties_as_json())