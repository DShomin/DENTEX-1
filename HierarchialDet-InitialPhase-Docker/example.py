import mmcv
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from mmengine import Config
import torch.nn.functional as F
config_file = '/content/drive/MyDrive/Colab Notebooks/DENTEX_MICCAI/MMdetection3/mmdetection/configs/custom/swintest.py'
cfg = Config.fromfile(config_file)
file = '/content/drive/MyDrive/Colab Notebooks/DENTEX_MICCAI/Submission/Valid/val_10.png'
img = mmcv.imread(file)
checkpoint_file = 'tutorial_exps/odontoai/Finetune/epoch_12.pth'
model = init_detector(cfg, checkpoint_file, device='cuda')
new_result = inference_detector(model, img)
pred = new_result.pred_instances.cpu().numpy()
Threshold = 0.7
CLASSES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48']
boxes = []
image_id = 50
detection = {
                  "name": "Regions of interest",
                  "type": "Multiple 2D bounding boxes",
                  "boxes": [],
                  "version": { "major": 1, "minor": 0 }}
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
  crop = img[bbox_r[1]:bbox_r[3],bbox_r[0]:bbox_r[2],:]
  cat1 = int(enum/10)-1
  cat2 = enum%10-1
  corners = [bbox[1], bbox[0], image_id], [bbox[1], bbox[2], image_id], [bbox[3], bbox[2], image_id], [bbox[3], bbox[0], image_id]
  #[x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
  output['name'] = str(cat1)+'-'+str(cat2)+'-'+'1'
  output['corners'] = corners
  output['probability'] = score
  boxes.append(output)
detection["boxes"] = boxes





