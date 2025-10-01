import os
from PIL import Image

class Edges2ShoesSplitter:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.edges_folder = os.path.join(output_folder, "A")
        self.shoes_folder = os.path.join(output_folder, "B")
        os.makedirs(self.edges_folder, exist_ok=True)
        os.makedirs(self.shoes_folder, exist_ok=True)

    def split_and_save(self):
        img_names = sorted([f for f in os.listdir(self.input_folder) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name in img_names:
            img_path = os.path.join(self.input_folder, img_name)
            img = Image.open(img_path)
            w, h = img.size
            mid = w // 2
            edges = img.crop((0, 0, mid, h))
            shoe = img.crop((mid, 0, w, h))
            edges.save(os.path.join(self.edges_folder, img_name))
            shoe.save(os.path.join(self.shoes_folder, img_name))

        print(f"Done! {len(img_names)} images split into 'A' (edges) and 'B' (shoes).")
