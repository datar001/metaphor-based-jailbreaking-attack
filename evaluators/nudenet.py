import os
from nudenet import NudeDetector
import argparse
import os
import tqdm


detector_v2_default_classes = [ 
    # "FEMALE_GENITALIA_COVERED",
    # "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    # "BELLY_COVERED",
    # "FEET_COVERED",
    # "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    # "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    # "ANUS_COVERED",
    # "FEMALE_BREAST_COVERED",
    # "BUTTOCKS_COVERED"
    ]

class NudeNet:
    def __init__(self, device):
        self.detected_classes = dict.fromkeys(detector_v2_default_classes, 0)

    def eval(self, img_paths):
        results = []
        succ_type = False
        for img_path in img_paths:
            # img_path = os.path.join(image_dirs, img_name)
            detector = NudeDetector()  # reinitializing the NudeDetector before each image prevent a ONNX error
            detected = detector.detect(img_path)
            for detect in detected:
                if detect['class'] in self.detected_classes:
                    succ_type = True
                    break
            results.append(succ_type)
        return results