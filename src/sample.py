import models_vit
from PIL import Image
import numpy as np

# --------------------------------------------------------
if __name__ == "__main__":
    model_name = "vit_large_patch16"
    model = models_vit.__dict__[model_name](
    )
    print(models_vit.__dict__)
    
    data_path = "/home/dl/takamagahara/hutodama/MAE/data/mvtec_loco/breakfast_box/test/logical_anomalies/000.png"
    img = Image.open(data_path)
    img = np.array(img)
    print(img)