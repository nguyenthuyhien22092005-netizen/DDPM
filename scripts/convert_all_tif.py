from PIL import Image
import os

ROOT = "C:/Users/Lenovo/PycharmProjects/PythonDiffusion"

def convert_tif_to_png(root):
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(subdir, file)
                png_path = tif_path.replace(".tif", ".png")

                try:
                    # Open and close automatically
                    with Image.open(tif_path) as img:
                        img.save(png_path, "PNG")

                    # Remove only AFTER closing
                    os.remove(tif_path)

                    print(f"Converted & removed: {png_path}")

                except Exception as e:
                    print(f"Error converting {tif_path}: {e}")

if __name__ == "__main__":
    convert_tif_to_png(ROOT)
