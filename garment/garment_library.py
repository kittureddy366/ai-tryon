GARMENT_PRESETS = {
    "tshirt_blue": {
        "label": "Blue T-Shirt",
        "mesh_rows": 12,
        "mesh_cols": 10,
        "cloth_color_bgra": (255, 255, 255, 245),
        "garment_obj_path": "uploads_files_3804428_Shirt.obj",
        "garment_zip_path": "22-t-shirt.zip",
        "texture_path": "png-clipart-t-shirt-polo-shirt-clothing-sleeve-black-t-shirt-black-crew-neck-t-shirt-tshirt-fashion-thumbnail.png",
        "physics": {"stiffness": 290.0, "damping": 0.989, "gravity": 720.0},
    },
    "tshirt_black": {
        "label": "Black T-Shirt",
        "mesh_rows": 12,
        "mesh_cols": 10,
        "cloth_color_bgra": (255, 255, 255, 245),
        "texture_path": "png-clipart-t-shirt-polo-shirt-clothing-sleeve-black-t-shirt-black-crew-neck-t-shirt-tshirt-fashion-thumbnail.png",
        "physics": {"stiffness": 320.0, "damping": 0.989, "gravity": 760.0},
    },
    "hoodie_red": {
        "label": "Red Hoodie",
        "mesh_rows": 14,
        "mesh_cols": 12,
        "cloth_color_bgra": (255, 255, 255, 245),
        "texture_path": "png-clipart-t-shirt-polo-shirt-clothing-sleeve-black-t-shirt-black-crew-neck-t-shirt-tshirt-fashion-thumbnail.png",
        "physics": {"stiffness": 350.0, "damping": 0.989, "gravity": 820.0},
    },
}


def list_preset_keys():
    return tuple(GARMENT_PRESETS.keys())


def get_preset(key):
    if key not in GARMENT_PRESETS:
        return GARMENT_PRESETS["tshirt_blue"]
    return GARMENT_PRESETS[key]
