import math
import torch

def SoftIoUnion(boxesA, boxesB):

    Acx, Acy, Aw, Ah = boxesA[0], boxesA[1], boxesA[2], boxesA[3]
    Bcx, Bcy, Bw, Bh = boxesB[0], boxesB[1], boxesB[2], boxesB[3]
    
    x1 = torch.max(Acx - Aw/2, Bcx - Bw/2)
    x2 = torch.min(Acx + Aw/2, Bcx + Bw/2)
    y1 = torch.max(Acy - Ah/2, Bcy - Bh/2)
    y2 = torch.min(Acy + Ah/2, Bcy + Bh/2)

    W_ab = torch.max(x2 - x1, torch.zeros(len(x2)))
    H_ab = torch.max(y2 - y1, torch.zeros(len(y2)))
    
    Sa = Aw * Ah
    Sb = Bw * Bh
    intersection = W_ab * H_ab
    union = Sa + Sb - intersection
    return intersection/union

class YoloLayer(torch.nn.Module):
    def __init__(self, anchors, stride, num_classes): 
        super().__init__()
        self.anchors=torch.FloatTensor(anchors)/stride
        self.stride=stride
        self.num_classes=num_classes

    def get_region_boxes(self, output, threshold):
        if output.dim() == 3: output = output.unsqueeze(0)  
        device = output.device # torch.device(torch_device)
        anchors = self.anchors.to(device)
        
        B, c, H, W = output.size()
        A = len(anchors)
        C = self.num_classes
        Ad = B*A*H*W

        assert c == (C+5)*A

        output = output.view(B*A, C+5, H*W).transpose(0,1).contiguous().view(C+5, Ad)

        Gx = torch.arange(0, W).repeat(B*A, H).view(Ad).to(device)
        Gy = torch.arange(0, H).repeat(W, 1).t().repeat(B*A, 1).view(Ad).to(device)
        ix = torch.LongTensor(range(0,2)).to(device)
        Aw = anchors.index_select(1, ix[0]).repeat(1, B, H*W).view(Ad)
        Ah = anchors.index_select(1, ix[1]).repeat(1, B, H*W).view(Ad)

        xs = torch.sigmoid(output[0]) + Gx
        ys = torch.sigmoid(output[1]) + Gy
        ws = torch.exp(output[2]) * Aw.detach()
        hs = torch.exp(output[3]) * Ah.detach()
        box_conf = torch.sigmoid(output[4])

        cls_conf = torch.nn.Softmax(dim=1)(output[5: C+5].transpose(0,1)).detach()
        cls_prob, cls_id = torch.max(cls_conf, 1)
        cls_prob = cls_prob.view(-1)
        cls_id = cls_id.view(-1)

        xs = xs.to('cpu')
        ys = ys.to('cpu')
        ws = ws.to('cpu')
        hs = hs.to('cpu')
        box_conf = box_conf.to('cpu') #, non_blocking=True for torch 4.1?
        cls_prob = cls_prob.to('cpu')
        cls_id = cls_id.to('cpu')


        boxes = [[] for i in range(B)]
        
        idx = torch.LongTensor(range(0,len(box_conf)))
        for i in idx[box_conf > threshold]:
            cx = xs[i]
            cy = ys[i]
            w = ws[i]
            h = hs[i]
            
            box = [cx/W, cy/H, w/W, h/H, box_conf[i], cls_prob[i], cls_id[i]]
            box = [i.item() for i in box]

            batch = int(i.item()/(A*H*W))
            boxes[batch].append(box)

        return boxes


    def build_targets(self, pred_boxes, target, anchors, H, W):
        ignore_threshold = 0.5

        # Works faster on CPU than on GPU.
        dev = torch.device('cpu')
        pred_boxes = pred_boxes.to(dev)
        target = target.to(dev)
        anchors = anchors.to(dev)

        B = target.size(0)
        A, As = anchors.size()
        Ad = A*H*W
        
        box_mask   = torch.zeros(B, A, H, W)
        conf_mask  = torch.ones (B, A, H, W)
        cls_mask   = torch.zeros(B, A, H, W)
        t_boxes    = torch.zeros(4, B, A, H, W)
        t_conf     = torch.zeros(B, A, H, W)
        t_cls      = torch.zeros(B, A, H, W)

        for b in range(B):
            p_box = pred_boxes[b*Ad: (b+1)*Ad].t()
            ious = torch.zeros(Ad)
            box = target[b].view(-1,5)

            for t in range(box.size(0)):
                if box[t][1] == 0:
                    break
                cx = box[t][0] * W
                cy = box[t][1] * H
                w  = box[t][2] * W
                h  = box[t][3] * H
                t_box = torch.FloatTensor([cx, cy, w, h]).repeat(Ad, 1).t()
                ious = torch.max(ious, SoftIoUnion(p_box, t_box))
            ignore = ious > ignore_threshold
            conf_mask[b][ignore.view(A, H, W)] = 0

            for t in range(box.size(0)):
                if box[t][1] == 0:
                    break
                    
                cx = box[t][0] * W
                cy = box[t][1] * H
                w  = (box[t][2] * W).float()
                h  = (box[t][3] * H).float()
                ci, cj = int(cx), int(cy)

                tmp_box = torch.FloatTensor([0, 0, w, h]).repeat(A, 1).t()
                a_box = torch.cat((torch.zeros(A, As), anchors), 1).t()
                _, best_a = torch.max( SoftIoUnion(tmp_box, a_box), 0)

                box_mask  [b][best_a][cj][ci] = 1
                conf_mask [b][best_a][cj][ci] = 1
                cls_mask  [b][best_a][cj][ci] = 1
                
                t_boxes[0][b][best_a][cj][ci] = cx - ci
                t_boxes[1][b][best_a][cj][ci] = cy - cj
                t_boxes[2][b][best_a][cj][ci] = math.log(w/anchors[best_a][0])
                t_boxes[3][b][best_a][cj][ci] = math.log(h/anchors[best_a][1])
                t_conf    [b][best_a][cj][ci] = 1
                t_cls     [b][best_a][cj][ci] = box[t][4]

        return box_mask, conf_mask, cls_mask, t_boxes, t_conf, t_cls


    def get_loss(self, output, target, return_single_value=True):
        device = output.device

        anchors = self.anchors.to(device)

        B, c, H, W = output.data.size()
        A = len(anchors)
        C = self.num_classes
        Ad = B*A*H*W

        output = output.view(B, A, (C+5), H, W)

        ix = torch.LongTensor(range(0,5)).to(device)
        p_boxes = output.index_select(2, ix[0:4]).view(B*A, -1, H*W).transpose(0,1).contiguous().view(4, Ad)
        
        p_boxes[0:2] = p_boxes[0:2].sigmoid()
        p_boxes[2:4] = p_boxes[2:4].exp()
        p_conf = output.index_select(2, ix[4]).view(B, A, H, W).sigmoid()

        Gx = torch.arange(0, W).repeat(B*A, H).view(Ad).to(device)
        Gy = torch.arange(0, H).repeat(W, 1).t().repeat(B*A, 1).view(Ad).to(device)
        Aw = anchors.index_select(1, ix[0]).repeat(1, B*H*W).view(Ad)
        Ah = anchors.index_select(1, ix[1]).repeat(1, B*H*W).view(Ad)

        pred_boxes = torch.FloatTensor(4, Ad).to(device)
        pred_boxes[0] = p_boxes[0] + Gx
        pred_boxes[1] = p_boxes[1] + Gy
        pred_boxes[2] = p_boxes[2] * Aw
        pred_boxes[3] = p_boxes[3] * Ah 
        pred_boxes = pred_boxes.transpose(0,1).contiguous().view(-1,4)

        box_mask, conf_mask, cls_mask, t_boxes, t_conf, t_cls = \
            self.build_targets(pred_boxes.detach(), target.detach(), anchors.detach(), H, W)

        Gc = torch.arange(5, C+5).long().to(device)
        p_cls  = output.index_select(2, Gc)
        p_cls  = p_cls.view(B*A, C, H*W).transpose(1,2).contiguous().view(Ad, C)
        cls_mask = (cls_mask == 1)
        t_cls = t_cls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1, C).to(device)
        p_cls = p_cls[cls_mask].view(-1, C)
        
        t_boxes = t_boxes.view(4, Ad).to(device)
        t_conf = t_conf.to(device)
        t_cls = t_cls.to(device)
        box_mask = box_mask.view(Ad).to(device)
        conf_mask = conf_mask.to(device)

        box_loss = torch.nn.MSELoss(reduction='sum')(p_boxes *box_mask, t_boxes *box_mask)
        conf_loss = torch.nn.MSELoss(reduction='sum')(p_conf *conf_mask, t_conf *conf_mask)
        cls_loss = torch.nn.CrossEntropyLoss(reduction='sum')(p_cls, t_cls) if p_cls.size(0) > 0 else 0
        loss = box_loss/2 + conf_loss + cls_loss

        if math.isnan(loss.item()):
            print(p_conf, t_conf)            
            raise ValueError('YoloLayer has isnan in loss')
        
        if return_single_value:
            return loss
        else:
            return [loss, box_loss/2, conf_loss, cls_loss]