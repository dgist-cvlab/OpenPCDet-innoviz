
import os
from symbol import pass_stmt
import tensorflow as tf
import numpy as np
from pathlib import Path
import random
import tempfile
import shutil
import time
from scipy.spatial import ConvexHull, Delaunay
import shapely

__GT_BOX_DTYPE__ = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('dx', np.float32),
    ('dy', np.float32),
    ('dz', np.float32),
    ('heading', np.float32),
    ('class', np.float32),
])

def convex_hull_intersection(p1, p2):
    """Compute area of two convex hull's intersection area.
    p1,p2 are a list of (x,y) tuples of hull vertices.
    return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def polygon_clip(subjectPolygon, clipPolygon):
    """Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return outputList

class IOUBox:    
    """
    source: https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    One diff is the counter clock wise rotation used for contour calculation. see 'def contour'
    """
    def __init__(self, x, y, dx, dy, heading):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.heading = heading
        self.box = self.contour()

    def contour(self):
        """
        If rotate_clockwise is True, positive angles are rotated clockwise
        * For real-world top view (x=north, y=west) rotation around z (from x to y) is ccw
        * For image coordinates top view (x=east, y=south) rotation around z (from x to y) is cw
        """        
        import shapely.geometry
        import shapely.affinity

        w = self.dx
        h = self.dy
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, -self.heading)
        return shapely.affinity.translate(rc, self.x, self.y)

    def intersection(self, other):
        return self.contour().intersection(other.contour())

    def intersection_area(self, other):
        return self.intersection(other).area

    @property
    def area(self):
        return self.dx * self.dy

    def iou(self, other):
        intersection_area = self.intersection_area(other)
        return intersection_area / (self.area + other.area - intersection_area + 1e-9)

    @staticmethod
    def from_numpy(nb):
        return IOUBox(x=nb['x'], y=nb['y'], dx=nb['dx'], dy=nb['dy'], heading=nb['heading'])
    
    @staticmethod
    def from_numpy_raw(nb):
        return IOUBox(x=nb[0], y=nb[1], dx=nb[3], dy=nb[4], heading=nb[6])

def tmp_dir(timestamp: int, name: str) -> Path:
    return Path(tempfile.gettempdir()) / f'{timestamp}_{name}'

def calc_xy_iou(gt_boxes: np.array, det_boxes: np.array):
    # https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    print(f'calculating metrics. {len(gt_boxes)} gt_boxes, {len(det_boxes)} detected boxes')
    # create a result matrix in size of inputs
    # results = np.empty((len(gt_boxes), len(det_boxes)), dtype=float)
    results = np.zeros((len(gt_boxes), len(det_boxes)+1), dtype=float)
    for ri, gt_box in enumerate(gt_boxes):
        for ci, det_box in enumerate(det_boxes):
            results[ri, ci] = (IOUBox.from_numpy_raw(gt_box)).iou(IOUBox.from_numpy_raw(det_box))
    
    # for each gt use iou of detection with max iou (best match)
    ious = np.max(results, axis=1)
    print(f'ious: {ious}')

    return ious

def calc_xy_iou2(gt_boxes: np.array, det_boxes: np.array):
    # https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python
    print(f'calculating metrics. {len(gt_boxes)} gt_boxes, {len(det_boxes)} detected boxes')
    # create a result matrix in size of inputs
    # results = np.empty((len(gt_boxes), len(det_boxes)), dtype=float)
    results = np.zeros((len(gt_boxes), len(det_boxes)+1), dtype=float)
    gt_boxes_list = []
    det_boxes_list = []
    for ri, gt_box in enumerate(gt_boxes):
        gt_boxes_list.append(IOUBox.from_numpy_raw(gt_box).box)
    for ci, det_box in enumerate(det_boxes):
        det_boxes_list.append(IOUBox.from_numpy_raw(det_box).box)
    gt_box_poly = shapely.ops.unary_union(gt_boxes_list)
    det_box_poly = shapely.ops.unary_union(det_boxes_list)
    
    # for each gt use iou of detection with max iou (best match)
    ious = np.max(results, axis=1)
    print(f'ious: {ious}')

    return ious

def eval_xyiou(det_gt_annos_list):
    all_xy_iou = []
    all_xy_iou2 = []
    for det_annos, gt_annos in det_gt_annos_list:
        det_boxes = det_annos['boxes_lidar']
        gt_boxes = gt_annos['gt_boxes_lidar'][:,0:7]
        xy_iou = calc_xy_iou(gt_boxes, det_boxes)
        xy_iou2 = calc_xy_iou(det_boxes, gt_boxes)
        all_xy_iou.extend(xy_iou)
        all_xy_iou2.extend(xy_iou2)
    eval_dict = {
        'xy_iou1_sum': np.sum(all_xy_iou),
        'xy_iou2_sum': np.sum(all_xy_iou2),
        'cnt1': len(all_xy_iou),
        'cnt2': len(all_xy_iou2),
    }
    return eval_dict

def eval_xyiou_multiprocessing(det_gt_annos_list):
    import multiprocessing
    from tqdm import tqdm 
    from functools import partial
    
    eval_xyiou_function = partial(eval_xyiou)
    num_workers = 32
    parted_list = np.array_split(np.asarray(det_gt_annos_list), num_workers)

    # eval_xyiou_function(det_gt_annos_list)
    sequence_infos = []
    with multiprocessing.Pool(num_workers) as p:
        sequence_infos = list(tqdm(p.imap(eval_xyiou_function, parted_list),
                                    total=len(parted_list)))

    xy_iou_sum = 0.0
    xy_iou_cnt = 0.0
    for seq_info in sequence_infos:
        xy_iou_sum += seq_info['xy_iou1_sum']
        xy_iou_cnt += seq_info['cnt1']

    xy_iou_sum2 = 0.0
    xy_iou_cnt2 = 0.0
    for seq_info in sequence_infos:
        xy_iou_sum2 += seq_info['xy_iou2_sum']
        xy_iou_cnt2 += seq_info['cnt2']

    xy_iou_gt = xy_iou_sum / xy_iou_cnt if xy_iou_cnt > 0.0 else 0.0
    xy_iou_pred = xy_iou_sum2 / xy_iou_cnt2 if xy_iou_cnt2 > 0.0 else 0.0
    eval_dict = {
        'xy_iou_gt': xy_iou_gt,
        'xy_iou_pred': xy_iou_pred,
        'xy_iou': (xy_iou_gt + xy_iou_pred) / 2.0,
    }
    return eval_dict

def process_single_sequence(sequence_file, save_path, sampled_interval, has_label=True, use_two_returns=True, update_info_only=False, do_semantic_label=False):
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]

    # print('Load record (sampled_interval=%d): %s' % (sampled_interval, sequence_name))
    if not sequence_file.exists():
        print('NotFoundError: %s' % sequence_file)
        return []

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    cur_save_dir = save_path / sequence_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    if do_semantic_label:
        sem_label_dir = os.path.join(cur_save_dir, 'sem_label')
        os.makedirs(sem_label_dir, exist_ok=True)
    pkl_file = cur_save_dir / ('%s.pkl' % sequence_name)

    sequence_infos = []
    if pkl_file.exists() and not do_semantic_label: # temporary do everything
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        sequence_infos_old = None
        if not update_info_only:
            print('Skip sequence since it has been processed before: %s' % pkl_file)
            return sequence_infos
        else:
            sequence_infos_old = sequence_infos
            sequence_infos = []

    for cnt, data in enumerate(dataset):
        if cnt % sampled_interval != 0:
            continue
        # print(sequence_name, cnt)
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        info = {}
        pc_info = {'num_features': 5, 'lidar_sequence': sequence_name, 'sample_idx': cnt}
        info['point_cloud'] = pc_info

        info['frame_id'] = sequence_name + ('_%03d' % cnt)
        info['metadata'] = {
            'context_name': frame.context.name,
            'timestamp_micros': frame.timestamp_micros
        }
        image_info = {}
        for j in range(5):
            width = frame.context.camera_calibrations[j].width
            height = frame.context.camera_calibrations[j].height
            image_info.update({'image_shape_%d' % j: (height, width)})
        info['image'] = image_info

        pose = np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        info['pose'] = pose

        if has_label:
            annotations = generate_labels(frame, pose=pose)
            info['annos'] = annotations

        if update_info_only and sequence_infos_old is not None:
            assert info['frame_id'] == sequence_infos_old[cnt]['frame_id']
            num_points_of_each_lidar = sequence_infos_old[cnt]['num_points_of_each_lidar']
        else:
            num_points_of_each_lidar = save_lidar_points(
                frame, cur_save_dir / ('%04d.npy' % cnt), use_two_returns=use_two_returns, do_semantic_label=do_semantic_label and has_label, ori_annotation=annotations,
            )
        info['num_points_of_each_lidar'] = num_points_of_each_lidar

        sequence_infos.append(info)

    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)

    print('Infos are saved to (sampled_interval=%d): %s' % (sampled_interval, pkl_file))
    return sequence_infos
