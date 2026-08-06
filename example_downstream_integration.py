"""
Example: Integrating Adaptive Super-Resolution into Downstream Projects
(Land Cover Classification, Water Change Detection, Building Extraction)
"""

import os
from PIL import Image
from sr_pipeline_service import RemoteSensingSREngine

def main():
    # 1. Initialize the Super-Resolution Engine once
    sr_engine = RemoteSensingSREngine(ckpt_dir="all_checkpoints")

    # 2. Simulate loading a low-resolution satellite image from a land cover or change detection dataset
    sample_img_path = "sample_lr_patch.png"
    
    # Create a dummy test image if sample doesn't exist
    if not os.path.exists(sample_img_path):
        import numpy as np
        dummy = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(dummy).save(sample_img_path)

    raw_lr_satellite_image = Image.open(sample_img_path)

    # 3. Super-resolve the image 4x with region-aware attention prior to feeding into downstream model
    enhanced_hr_image, importance_map = sr_engine.enhance_pil_image(raw_lr_satellite_image, return_importance=True)

    print(f"Original LR Image Size: {raw_lr_satellite_image.size}")
    print(f"Super-Resolved HR Image Size: {enhanced_hr_image.size}")
    
    # Save output ready for Land Cover / Change Detection model
    enhanced_hr_image.save("super_resolved_for_downstream_model.png")
    importance_map.save("importance_heatmap.png")
    
    print("\n✅ Enhanced high-resolution image saved successfully!")
    print("Now pass 'enhanced_hr_image' directly to your Land Cover / Change Detection model!")

if __name__ == "__main__":
    main()
