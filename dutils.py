import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, ToPILImage

def toTensor(image, size):
    image_ = image.resize(size)
    x = ToTensor()(image_).unsqueeze(0)
    return x

def IoUnion(boxA, boxB):
    Acx, Acy, Aw, Ah = boxA[:4]
    Bcx, Bcy, Bw, Bh = boxB[:4]
    
    x1 = max(Acx - Aw/2, Bcx - Bw/2)
    x2 = min(Acx + Aw/2, Bcx + Bw/2)
    y1 = max(Acy - Ah/2, Bcy - Bh/2)
    y2 = min(Acy + Ah/2, Bcy + Bh/2)

    w_cross = max(x2 - x1, 0)
    h_cross = max(y2 - y1, 0)

    Sa = Aw * Ah
    Sb = Bw * Bh
    intersection = w_cross * h_cross
    union = Sa + Sa - intersection
    iou = intersection/union
    return iou

def NonMaxSuppression(boxes, IoU=0.5):
    if len(boxes) == 0:
        return boxes

    confs = [(1-b[4]) for b in boxes]
    sorted_idx = np.argsort(confs)
    preds = []

    for i in range(len(boxes)):
        box_i = boxes[sorted_idx[i]]
        if confs[i] > -1:
            preds.append(box_i)
            for j in range(i+1, len(boxes)):
                if confs[j] > -1:
                    box_j = boxes[sorted_idx[j]]
                    if IoUnion(box_i, box_j) > IoU:
                        confs[j] = -1
    return preds

def YoloHead(preds):
    preds = np.array(preds)
    # Select Boxes, Scores and Classes
    boxes = preds[:, :4]
    scores = preds[:, 4]
    classes = preds[:, -1].astype('int')
    # Boxes to Corners
    cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    boxes = np.stack((x1, y1, x2, y2), axis=1)
    
    return scores, boxes, classes

def DrawBoxes(img, scores, boxes, classes, dataset='coco'):
    # Copy Image (to exclude changes in the original image)
    image = img.copy()
    W, H = image.size
    scale = np.array([W, H, W, H])
    thickness = (W + H)//300
    # Import Coco Classes, Font and Colors
    font = ImageFont.truetype(font='data/font/FiraMono-Medium.otf', size=int(0.03*H + 0.5))
    class_names = [c.strip() for c in open('data/'+dataset+'_classes.txt').readlines()]
    colors = np.load('data/'+dataset+'_colors.npy')
    # Cycle through the Classes
    for i, c in reversed(list(enumerate(classes))):
        color = colors[c]
        cls = class_names[c]
        box = boxes[i]
        score = scores[i]
        # Box`s Label
        label = '{} {:.2f}'.format(cls, score)
        # Draw Image
        draw = ImageDraw.Draw(image)
        size = draw.textsize(label, font)
        # Choose Frontiers
        left, top, right, bottom = box *scale
        # Exclude out of the image situations
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(W, int(right))
        bottom = min(H, int(bottom))

        if top - size[1] >= 0:
            origin = np.array([left, top - size[1]])
        else:
            origin = np.array([left, top + 1])
            
        # Draw Frame
        for k in range(thickness):
            draw.rectangle([left + k, top + k, right - k, bottom - k], outline=tuple(color))
        # Draw Label
        draw.rectangle([tuple(origin), tuple(origin + size)], fill=tuple(color))
        draw.text(origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image

def PictureDetection(model, img, size=(416,416), threshold=.25, IoU=.5, dataset='coco'):
    if type(img) is str:
        image = Image.open(img).convert('RGB')
        x = toTensor(image, size)
    else:
        image = ToPILImage()(img)
        x = ToTensor()(image.resize(size))
    preds = model.predict(x, threshold)[0]
    if len(preds) != 0:
        preds = NonMaxSuppression(preds, IoU)
        scores, boxes, classes = YoloHead(preds)
        image = DrawBoxes(image, scores, boxes, classes, dataset)
    return image

def VideoDetection(model, img, size=(416,416), threshold=.25, IoU=.5, dataset='coco'):
    image = ToPILImage()(img)
    x = ToTensor()(image.resize(size))
    preds = model.predict(x, threshold)[0]
    preds = NonMaxSuppression(preds, IoU)
    if len(preds) != 0:
        scores, boxes, classes = YoloHead(preds)
        image = DrawBoxes(image, scores, boxes, classes, dataset)
    return image