from pycocotools.cocoeval import COCOeval
import json
import torch

from projects.common.enums import RunMode
import projects.common.constants as C


def evaluate_coco(dataset, model, threshold=0.05, device=None):
    model.eval()
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            elem = dataset.df_data.iloc[index]
            image_id = elem['img_id']
            img_path = elem['img_path']

            # run network
            if device:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).to(device).unsqueeze(dim=0), mode="inference")
            elif torch.cuda.is_available():
                # input: (B, 3, 480, 640) / output: (B, 57600, 4), (B, 57600, 8), (B, 57600, 4)
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0), mode="inference")  
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0), mode="inference")
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale
            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'id'          : box_id,
                        'image_id'    : int(image_id),
                        'category_id' : dataset.label_to_coco_label(label),
                        'segmentation': [],
                        'area'        : int(0),
                        'bbox'        : box.tolist(),
                        'iscrowd'     : int(0),
                        'attributes'  : {
                            "occluded": False,
                            "rotation": 0.0
                        }
                    }
                    
                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(image_id)

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')
        
        # if not len(results):
        #     return

        # write output
        with open(C.ANSWER_SAMPLE, 'r') as json_file:
            json_data = json.load(json_file)
            json_data.pop("annotations")
        json_data["annotations"] = results
        json.dump(json_data, open('{}_bbox_results.json'.format(dataset.mode.value), 'w'), indent=4)

        # load results in COCO evaluation tool
        # coco_true = dataset.coco
        # coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.mode..value))

        # run COCO evaluation
        # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        # coco_eval.params.imgIds = image_ids
        # coco_eval.evaluate()
        # coco_eval.accumulate()
        # coco_eval.summarize()

        model.train()

        return