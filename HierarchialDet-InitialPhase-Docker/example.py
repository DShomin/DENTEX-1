import mmcv
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from mmengine import Config
import torch.nn.functional as F

config_file = './configs/swintest.py'
cfg = Config.fromfile(config_file)
file = './test/val_15.png'
img = mmcv.imread(file)
checkpoint_file = './configs/epoch_12.pth'
model = init_detector(cfg, checkpoint_file, device='cuda')
new_result = inference_detector(model, img)
pred = new_result.pred_instances.cpu().numpy()
Threshold = 0.7
CLASSES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48']

for i, score in enumerate(pred.scores[pred.scores > Threshold]):
  print('socre:', score)
  enum = pred.labels[i]
  enum = CLASSES[enum-1]
  print('enum:', enum)
  # mask = pred.masks[i,:,:]*img[:,:,0]
  bbox = pred.bboxes[i]
  bbox = [round(i) for i in bbox]
  crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2],:]






