import argparse
import glob
import json

import matplotlib.pyplot as plt
import mmcv
import SimpleITK as sitk
import torch.nn.functional as F
from mmdet.apis import inference_detector, init_detector
from mmengine import Config

list_ids = [
                          {"height": 1316, "width": 2892, "id": 1, "file_name": "val_15.png"},
                          {"height": 1316, "width": 2942, "id": 2, "file_name": "val_38.png"},
                          {"height": 1316, "width": 2987, "id": 3, "file_name": "val_33.png"},
                          {"height": 1504, "width": 2872, "id": 4, "file_name": "val_30.png"},
                          {"height": 1316, "width": 2970, "id": 5, "file_name": "val_5.png"},
                          {"height": 1316, "width": 2860, "id": 6, "file_name": "val_21.png"},
                          {"height": 1504, "width": 2804, "id": 7, "file_name": "val_39.png"},
                          {"height": 1316, "width": 2883, "id": 8, "file_name": "val_46.png"},
                          {"height": 1316, "width": 2967, "id": 9, "file_name": "val_20.png"},
                          {"height": 1504, "width": 2872, "id": 10, "file_name": "val_3.png"},
                          {"height": 1316, "width": 2954, "id": 11, "file_name": "val_29.png"},
                          {"height": 976, "width": 1976, "id": 12, "file_name": "val_2.png"},
                          {"height": 1316, "width": 2870, "id": 13, "file_name": "val_16.png"},
                          {"height": 1316, "width": 3004, "id": 14, "file_name": "val_25.png"},
                          {"height": 1316, "width": 2745, "id": 15, "file_name": "val_24.png"},
                          {"height": 1504, "width": 2872, "id": 16, "file_name": "val_31.png"},
                          {"height": 1316, "width": 2782, "id": 17, "file_name": "val_26.png"},
                          {"height": 1316, "width": 2744, "id": 18, "file_name": "val_44.png"},
                          {"height": 1504, "width": 2872, "id": 19, "file_name": "val_27.png"},
                          {"height": 1504, "width": 2868, "id": 20, "file_name": "val_41.png"},
                          {"height": 1316, "width": 3000, "id": 21, "file_name": "val_37.png"},
                          {"height": 1316, "width": 2797, "id": 22, "file_name": "val_40.png"},
                          {"height": 1316, "width": 2930, "id": 23, "file_name": "val_6.png"},
                          {"height": 1316, "width": 3003, "id": 24, "file_name": "val_18.png"},
                          {"height": 1316, "width": 2967, "id": 25, "file_name": "val_13.png"},
                          {"height": 1316, "width": 2822, "id": 26, "file_name": "val_8.png"},
                          {"height": 1316, "width": 2836, "id": 27, "file_name": "val_49.png"},
                          {"height": 1316, "width": 2704, "id": 28, "file_name": "val_23.png"},
                          {"height": 976, "width": 1976, "id": 29, "file_name": "val_1.png"},
                          {"height": 1504, "width": 2872, "id": 30, "file_name": "val_43.png"},
                          {"height": 1504, "width": 2872, "id": 31, "file_name": "val_28.png"},
                          {"height": 1504, "width": 2872, "id": 32, "file_name": "val_19.png"},
                          {"height": 1316, "width": 2728, "id": 33, "file_name": "val_14.png"},
                          {"height": 1316, "width": 2747, "id": 34, "file_name": "val_32.png"},
                          {"height": 976, "width": 1976, "id": 35, "file_name": "val_36.png"},
                          {"height": 1316, "width": 2829, "id": 36, "file_name": "val_47.png"},
                          {"height": 1316, "width": 2846, "id": 37, "file_name": "val_48.png"},
                          {"height": 1536, "width": 3076, "id": 38, "file_name": "val_17.png"},
                          {"height": 976, "width": 1976, "id": 39, "file_name": "val_42.png"},
                          {"height": 1504, "width": 2884, "id": 40, "file_name": "val_45.png"},
                          {"height": 1316, "width": 2741, "id": 41, "file_name": "val_9.png"},
                          {"height": 1316, "width": 2794, "id": 42, "file_name": "val_4.png"},
                          {"height": 1316, "width": 2959, "id": 43, "file_name": "val_34.png"},
                          {"height": 1316, "width": 2874, "id": 44, "file_name": "val_10.png"},
                          {"height": 1316, "width": 2978, "id": 45, "file_name": "val_35.png"},
                          {"height": 1504, "width": 2884, "id": 46, "file_name": "val_11.png"},
                          {"height": 1316, "width": 2794, "id": 47, "file_name": "val_12.png"},
                          {"height": 1316, "width": 2959, "id": 48, "file_name": "val_7.png"},
                          {"height": 1316, "width": 2912, "id": 49, "file_name": "val_22.png"},
                          {"height": 1504, "width": 2872, "id": 50, "file_name": "val_0.png"},
                      ]


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--nclass",
        type=int,
        default=3,
        help="Number of trained classes",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Hierarchialdet:
    def __init__(self):
        self.cfg = None
        self.demo = None
        self.model = None
        self.input_dir = "input"

    def load_model(self, config_path, model_path):
        cfg = Config.fromfile(config_path)
        self.model = init_detector(cfg, model_path, device='cuda')
        print("model loaded from {} and {}".format(config_path, model_path))

    def setup(self):
        self.load_model(config_path="/opt/app/configs/swintest.py", model_path="/opt/app/configs/epoch_12.pth")
        print("setup")
        

    def process(self):
        self.setup()

        file_path = glob.glob('/input/images/panoramic-dental-xrays/*.mha')[0]
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        print("test..")
        detection = {
                "name": "Regions of interest",
                "type": "Multiple 2D bounding boxes",
                "boxes": [],
                "version": { "major": 1, "minor": 0 }}    
        boxes = []
        for image_idx in range(image_array.shape[2]):
            image_name = "val_{}.png".format(image_idx)
            
            for input_img in list_ids:
                if input_img["file_name"] == image_name:
                    img_id = input_img["id"]
                    k_boxes = self.run_mmdetection(image_array[:,:,image_idx,:], img_id)
                    boxes.extend(k_boxes)
            
        detection["boxes"] = boxes

        output_file = "/output/abnormal-teeth-detection.json"
        with open(output_file, "w") as f:
            json.dump(detection, f)

        print("Inference completed. Results saved to", output_file)

    def run_mmdetection(self, image, image_id):
        
        img = mmcv.imread(image)
        new_result = inference_detector(self.model, img)
        pred = new_result.pred_instances.cpu().numpy()
        Threshold = 0.7
        CLASSES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48']
        boxes = []

        for i, score in enumerate(pred.scores[pred.scores > Threshold]):
            output = {}
            enum = pred.labels[i]
            if enum-1 < 32:
                enum = CLASSES[enum-1]
            else:
                continue
            enum = int(enum)
            bbox = pred.bboxes[i]
            x1, y1, x2, y2 = bbox
            # bbox_r = [round(i) for i in bbox]
            cat1 = int(enum/10)-1
            cat2 = enum%10-1
            # corners = [[float(bbox[1]), float(bbox[0]), image_id], [float(bbox[1]), float(bbox[2]), image_id], [float(bbox[3]), float(bbox[2]), image_id], [float(bbox[3]), float(bbox[0]), image_id]]
            corners = [
                [float(x1), float(y1), image_id],
                [float(x1), float(y2), image_id],
                [float(x2), float(y2), image_id],
                [float(x2), float(y1), image_id]
            ]
            
            # [x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
            output['name'] = str(cat1)+'-'+str(cat2)+'-'+'1'
            output['corners'] = corners
            output['probability'] = float(score)
            boxes.append(output)
        return boxes
       
if __name__ == "__main__":
    Hierarchialdet().process()
