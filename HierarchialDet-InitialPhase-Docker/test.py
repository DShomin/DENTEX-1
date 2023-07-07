import json
import glob
import mmcv
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from mmengine import Config
import torch.nn.functional as F

def run_mmdetection(image,image_id):
    config_file = './configs/swintest.py'
    cfg = Config.fromfile(config_file)
    img = mmcv.imread(image)
    checkpoint_file = './configs/epoch_12.pth'
    model = init_detector(cfg, checkpoint_file, device='cpu')
    new_result = inference_detector(model, img)
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
        bbox_r = [round(i) for i in bbox]
        cat1 = int(enum/10)-1
        cat2 = enum%10-1
        corners = [[float(bbox[1]), float(bbox[0]), image_id], [float(bbox[1]), float(bbox[2]), image_id], [float(bbox[3]), float(bbox[2]), image_id], [float(bbox[3]), float(bbox[0]), image_id]]
        #[x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
        output['name'] = str(cat1)+'-'+str(cat2)+'-'+'1'
        output['corners'] = corners
        output['probability'] = score
        boxes.append(output)
    return boxes

def test():
    print("test..")
    image_name = "val_15.png"
    file = '/test/{}'.format(image_name)
    img = mmcv.imread(file)
    k_boxes = run_mmdetection(img,1)
    detection = {
                "name": "Regions of interest",
                "type": "Multiple 2D bounding boxes",
                "boxes": [],
                "version": { "major": 1, "minor": 0 }} 
    boxes = []
    boxes.append(k_boxes)
            
    detection["boxes"] = boxes
    output_file = "/output/abnormal-teeth-detection.json"
    print(detection)
    with open(output_file, "w") as f:
        json.dump(str(detection), f)


if __name__ == "__main__":
    test()