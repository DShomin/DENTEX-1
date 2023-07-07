import json

all_outputs = []
img_ids = []
list_ids = [{"height": 1316, "width": 2892, "id": 1, "file_name": "val_15.png"}]

print("test..")
image_name = "val_15.png"
file = '/test/{}'.format(image_name)
img = mmcv.imread(file)
predictions, _ = run(img)
instances = predictions["instances"]
all_outputs.append(instances)
for input_img in list_ids:
    if input_img["file_name"] == image_name:
        img_id = input_img["id"]
img_ids.append(img_id)
coco_annotations = custom_format_output(all_outputs,img_ids)

output_file = "/output/abnormal-teeth-detection.json"
with open(output_file, "w") as f:
    json.dump(coco_annotations, f)