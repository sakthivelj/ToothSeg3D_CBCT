import pandas as pd
from matplotlib import colors

def get_colormap(file_path : str ='csv_files/colormap_slicer.csv', num_classes=32) -> colors.ListedColormap:
    tooth_colors = pd.read_csv(file_path, delimiter=";", header=None)
    tooth_colors_df = tooth_colors.iloc[:,2:5]
    tooth_colors_df.columns = ['r','g','b']
    slicer_colorspace = tooth_colors_df.to_numpy()/255
    slicer_map = colors.ListedColormap(slicer_colorspace[:num_classes+1], 'slicer_colors')
    return slicer_map