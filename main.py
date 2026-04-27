import cv2 as cv
import os
import numpy as np
import torch
import math
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression
from cellpose import models
from scipy.interpolate import splprep, splev

def refine_contour(cnt, epsilon_factor=0.005):
    """Douglas-Peucker算法简化轮廓，保留关键顶点"""
    epsilon = epsilon_factor * cv.arcLength(cnt, True)
    return cv.approxPolyDP(cnt, epsilon, True)

def spline_smooth(cnt, num_points=800, smooth=1.0):
    """B样条平滑轮廓，减少噪声干扰"""
    cnt = cnt.squeeze().astype(float)
    tck, u = splprep(cnt.T, s=smooth, per=1)
    u_new = np.linspace(u.min(), u.max(), num_points)
    return np.array(splev(u_new, tck)).T.reshape(-1, 1, 2).astype(int)

def find_contour_endpoints(contour):
    """取X坐标最小（最左）和最大（最右）的点作为端点"""
    points = contour.squeeze()
    if len(points) < 2:
        return []
    sorted_by_x = sorted(points, key=lambda p: p[0])
    left_point = sorted_by_x[0]
    right_point = sorted_by_x[-1]
    return [left_point, right_point]

def divide_quadrants(e1, e2):
    """基于两个端点划分四个象限，适配OpenCV y轴向下的坐标系"""
    midpoint = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
    line_vec = (e2[0] - e1[0], e2[1] - e1[1])
    up_dir = (line_vec[1], -line_vec[0])

    def get_quadrant(point):
        point_to_mid = (point[0] - midpoint[0], point[1] - midpoint[1])
        is_left = np.dot(point_to_mid, line_vec) < 0
        is_above = np.dot(point_to_mid, up_dir) > 0
        if is_left and is_above:
            return 1
        elif not is_left and not is_above:
            return 3
        else:
            return 0
    return get_quadrant

def point_to_line_distance(point, line_p1, line_p2):
    """计算点到直线的垂直距离"""
    a = line_p2[1] - line_p1[1]
    b = line_p1[0] - line_p2[0]
    c = line_p2[0] * line_p1[1] - line_p1[0] * line_p2[1]
    return abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)

def find_polygon_concave_points(contour_poly, step=2, concave_angle_thresh=160):
    """检测多边形轮廓的凹点（角度大于阈值的顶点）"""
    points = contour_poly.squeeze()
    n = len(points)
    concave_points = []
    if n < 2 * step + 1:
        print(f"Contour vertices insufficient ({n}), step auto-adjusted to 1")
        step = 1
    for i in range(n):
        p_prev = points[(i - step) % n]
        p_curr = points[i]
        p_next = points[(i + step) % n]
        vec_prev = p_prev - p_curr
        vec_next = p_next - p_curr
        vec_prev_3d = np.hstack([vec_prev, 0])
        vec_next_3d = np.hstack([vec_next, 0])
        cross = np.cross(vec_prev_3d, vec_next_3d)[2]
        dot = np.dot(vec_prev, vec_next)
        angle = math.degrees(math.atan2(abs(cross), dot))

        if cross < 0:
            angle = 360 - angle
        if angle > concave_angle_thresh:
            concave_points.append((i, p_curr))
            print(f"Concave point: index {i}, angle {angle:.1f}°, coords {p_curr}")
    return concave_points

def calculate_concave_depth(point, p_prev, p_next):
    """计算凹点深度：点到前后点连线的垂直距离"""
    return point_to_line_distance(point, p_prev, p_next)

def find_quadrant_depressions(contour, endpoints, step=2, concave_angle_thresh=160,
                              max_mid_dist=50, max_dist_diff=20):
    """找最优融合点"""
    if len(endpoints) != 2:
        print("Insufficient endpoints (need 2), cannot divide quadrants")
        return []
    e1, e2 = endpoints
    print(f"Endpoints confirmed: left {e1}, right {e2}")
    endpoints_mid = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
    get_quadrant = divide_quadrants(e1, e2)
    contour_poly = refine_contour(contour)
    print(f"Polygon simplified vertices: {len(contour_poly)}")
    if len(contour_poly) < 4:
        print("Contour too simple, cannot detect concave points")
        return []
    concave_points = find_polygon_concave_points(
        contour_poly, step=step, concave_angle_thresh=concave_angle_thresh )
    if len(concave_points) < 2:
        print(f"Insufficient concave points ({len(concave_points)}), try step=1")
        concave_points = find_polygon_concave_points(contour_poly, step=1, concave_angle_thresh=160)
        if len(concave_points) < 2:
            return []
    quad1_points = []
    quad3_points = []
    poly_points = contour_poly.squeeze()
    for idx, p in concave_points:
        quad = get_quadrant(p)
        print(f"Concave point coords {p}, calculated quadrant={quad}")
        if quad not in (1, 3):
            continue
        p_prev = poly_points[(idx - step) % len(poly_points)]
        p_next = poly_points[(idx + step) % len(poly_points)]
        depth = calculate_concave_depth(p, p_prev, p_next)
        if quad == 1:
            quad1_points.append((depth, p, idx))
        else:
            quad3_points.append((depth, p, idx))
        print(f"Quadrant {quad} concave point: coords {p}, depth {depth:.2f}")

    if len(quad1_points) == 0 or len(quad3_points) == 0:
        print(
            f"Insufficient concave points in target quadrants (quad1: {len(quad1_points)}, quad3: {len(quad3_points)})")
        return []
    valid_pairs = []
    for q1 in quad1_points:
        d1_depth, d1_p, _ = q1
        for q3 in quad3_points:
            d2_depth, d2_p, _ = q3
            fusion_mid = ((d1_p[0] + d2_p[0]) / 2, (d1_p[1] + d2_p[1]) / 2)
            mid_dist = math.hypot(fusion_mid[0] - endpoints_mid[0], fusion_mid[1] - endpoints_mid[1])
            dist1 = point_to_line_distance(d1_p, e1, e2)
            dist2 = point_to_line_distance(d2_p, e1, e2)
            dist_diff = abs(dist1 - dist2)
            if mid_dist <= max_mid_dist and dist_diff <= max_dist_diff:
                valid_pairs.append((mid_dist, dist_diff, d1_depth + d2_depth, d1_p, d2_p))

    if valid_pairs:
        valid_pairs.sort(key=lambda x: (x[0], x[1], -x[2]))
        best_pair = valid_pairs[0]
        print(
            f"Best fusion pair: distance to endpoint mid {best_pair[0]:.2f}px, distance diff {best_pair[1]:.2f}px, total depth {best_pair[2]:.2f}")
        return [best_pair[3], best_pair[4]]
    else:
        print(f"No valid fusion pairs, select deepest points in each quadrant")
        quad1_deepest = max(quad1_points, key=lambda x: x[0])
        quad3_deepest = max(quad3_points, key=lambda x: x[0])
        return [quad1_deepest[1], quad3_deepest[1]]

def find_four_target_points(contours, roi_shape, frame_count, roi_x1, roi_y1,
                            step=2, concave_angle_thresh=160, max_mid_dist=50, max_dist_diff=20):
    """寻找关键点函数，返回：关键点列表, 动态key阈值, 动态endpoint阈值, 动态mid阈值, 动态diff阈值"""
    roi_h, roi_w = roi_shape
    roi_area = roi_h * roi_w
    target_points = []
    valid_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if 0.05 * roi_area <= area <= 0.9 * roi_area:
            smoothed_cnt = spline_smooth(cnt)
            valid_contours.append(smoothed_cnt)
    # 动态阈值计算(防止意外给了初值)
    dynamic_key_thresh = 35.0
    dynamic_end_thresh = 35.0
    dynamic_mid_dist = 20.0
    dynamic_dist_diff = 8.0
    target_length = 0.0
    length_type = "default"

    k_key = 0.1
    k_end = 0.1
    k_mid = 0.06
    k_diff = 0.025

    if len(valid_contours) == 1:
        cnt = valid_contours[0]
        try:
            ellipse = cv.fitEllipse(cnt)
            (x, y), (minor_axis, major_axis), angle = ellipse
            target_length = major_axis
            length_type = "ellipse_major"
        except:
            rect = cv.minAreaRect(cnt)
            w, h = rect[1]
            target_length = np.sqrt(w ** 2 + h ** 2)
            length_type = "rect_diagonal"
        dynamic_key_thresh = np.clip(target_length * k_key, 10, 80)
        dynamic_end_thresh = np.clip(target_length * k_end, 10, 80)
        dynamic_mid_dist = np.clip(target_length * k_mid, 5, 40)
        dynamic_dist_diff = np.clip(target_length * k_diff, 3, 15)
        print(f"✅ 动态阈值 | 目标长度({length_type})={target_length:.0f}px | "
              f"Key={dynamic_key_thresh:.1f} | Endpoint={dynamic_end_thresh:.1f}")

    if len(valid_contours) == 1:
        print(f"Detected 1 valid contour, enable single-contour fusion point logic")
        cnt = valid_contours[0]
        endpoints = find_contour_endpoints(cnt)
        if len(endpoints) != 2:
            print(f"Endpoint detection failed ({len(endpoints)} points), skip")
            return target_points, dynamic_key_thresh, dynamic_end_thresh, dynamic_mid_dist, dynamic_dist_diff
        depression_points = find_quadrant_depressions(
            cnt, endpoints, step=step, concave_angle_thresh=concave_angle_thresh,
            max_mid_dist=dynamic_mid_dist, max_dist_diff=dynamic_dist_diff
        )

        if len(depression_points) == 2:
            target_points = endpoints + depression_points
            target_points = [tuple(map(int, p)) for p in target_points]
            e1, e2, d1, d2 = target_points
            print(f"\n[Frame {frame_count}] 找到4个关键点:")
            print(f"左端点: {e1} | 右端点: {e2} | 融合点1: {d1} | 融合点2: {d2}\n")
        else:
            target_points = [tuple(map(int, p)) for p in endpoints]
            print(f"\n[Frame {frame_count}] 仅找到2个端点: {target_points}\n")
        return target_points, dynamic_key_thresh, dynamic_end_thresh, dynamic_mid_dist, dynamic_dist_diff
    # 无有效轮廓
    else:
        print(f"有效轮廓数量={len(valid_contours)}，跳过")
        return target_points, dynamic_key_thresh, dynamic_end_thresh, dynamic_mid_dist, dynamic_dist_diff

# ---------------------- YOLOv5辅助函数 ----------------------
def letterbox(im, new_shape=(640, 640), stride=32, color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (math.ceil(shape[1] * r / stride) * stride, math.ceil(shape[0] * r / stride) * stride)
    dw, dh = (new_shape[1] - new_unpad[0]) // 2, (new_shape[0] - new_unpad[1]) // 2
    im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR) if shape != new_unpad[::-1] else im
    return cv.copyMakeBorder(im, dh, dh, dw, dw, cv.BORDER_CONSTANT, value=color), r, (dw, dh)

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] = (coords[:, [0, 2]] - pad[0]) / gain
    coords[:, [1, 3]] = (coords[:, [1, 3]] - pad[1]) / gain
    return np.clip(coords.round().astype(int), 0, [img0_shape[1], img0_shape[0], img0_shape[1], img0_shape[0]])

def calculate_iou(roi1, roi2):
    x1_1, y1_1, x2_1, y2_1 = roi1
    x1_2, y1_2, x2_2, y2_2 = roi2
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)
    return iou

def calculate_points_change(current_points, prev_points):
    if len(current_points) < 2 or len(prev_points) < 2:
        return 0.0, 0.0, 0
    min_len = min(len(current_points), len(prev_points))
    dists = []
    for i in range(min_len):
        curr_p = current_points[i]
        prev_p = prev_points[i]
        dist = math.hypot(curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
        dists.append(dist)
    valid_count = len(dists)
    max_dist = max(dists) if valid_count > 0 else 0.0
    avg_dist = sum(dists) / valid_count if valid_count > 0 else 0.0
    return max_dist, avg_dist, valid_count

def calculate_endpoint_change(current_points, prev_points):
    if len(current_points) < 2 or len(prev_points) < 2:
        return 0.0, 0.0
    endpoint_dists = []
    for i in [0, 1]:
        curr_p = current_points[i]
        prev_p = prev_points[i]
        dist = math.hypot(curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
        endpoint_dists.append(dist)
    return max(endpoint_dists), sum(endpoint_dists) / 2

def draw_key_frame_label(frame, max_dist, endpoint_max_dist, key_frame_thresh, endpoint_change_thresh):
    cv.rectangle(frame, (10, 10), (320, 80), (0, 0, 255), -1)
    cv.putText(frame, "KEY FRAME (Backtrack)", (20, 35),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 3)
    cv.putText(frame, "KEY FRAME (Backtrack)", (20, 35),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame, f"Max Dist: {max_dist:.1f}px", (20, 55),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv.putText(frame, f"Endpoint Dist: {endpoint_max_dist:.1f}px", (20, 75),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# ---------------------- 视频处理主函数 ----------------------
def main_process(video_path, model_path, save_dir,
                 step=1, concave_angle_thresh=180,
                 save_key_frames_separately=True, iou_match_thresh=0.3,
                 max_backtrack_frames=2):
    os.makedirs(save_dir, exist_ok=True)
    key_frame_dir = os.path.join(save_dir, "key_frames")
    if save_key_frames_separately:
        os.makedirs(key_frame_dir, exist_ok=True)
    orig_frame_dir = os.path.join(save_dir, "original_frames")
    os.makedirs(orig_frame_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "four_points_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "Frame_ID,Left_x,Left_y,Right_x,Right_y,Fusion1_x,Fusion1_y,Fusion2_x,Fusion2_y,Is_Key_Frame,"
            "Max_Dist(px),Avg_Dist(px),Endpoint_Max_Dist(px),Endpoint_Avg_Dist(px),"
            "Compare_Type(Adjacent/Cross),ROI_Match_Status,Dynamic_Key_Thresh,Dynamic_Endpoint_Thresh\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from models.yolo import Detect
    torch.serialization.add_safe_globals([Detect])
    model_yolo = attempt_load(model_path, device=device)
    model_yolo.eval()
    STRIDE = int(model_yolo.stride.max())#加载YOLOv5

    model_cellpose = models.CellposeModel(
        pretrained_model="weights/cellpose-sam",
        device=device
    )#加载cellpose
    cap = cv.VideoCapture(video_path)
    orig_w, orig_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_time, frame_count = 0, 0
    print(f"Video resolution: {orig_w}x{orig_h}")
    frame_cache = []
    key_frame_set = set()
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_time = time.perf_counter()
            frame_copy = frame.copy()
            orig_frame_path = os.path.join(orig_frame_dir, f"orig_frame_{frame_count}.jpg")
            cv.imwrite(orig_frame_path, frame)

            current_max_dist = 0.0
            current_avg_dist = 0.0
            current_endpoint_max_dist = 0.0
            current_endpoint_avg_dist = 0.0
            is_key_frame = False
            compare_type = "No_Compare"
            roi_match_status = "No_Match"
            curr_dynamic_key = 35.0
            curr_dynamic_end = 35.0

            # YOLOv5检测
            img_resized, ratio, (pad_w, pad_h) = letterbox(frame, (640, 640), STRIDE)
            img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            pred = model_yolo(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

            curr_roi_info = []
            if pred and len(pred[0]) > 0:
                dets = pred[0].cpu().numpy()
                dets[:, :4] = scale_coords(img_resized.shape[:2], dets[:, :4], (orig_h, orig_w))
                print(f"\n[Frame {frame_count}] Detected {len(dets)} ROIs")
                for det in dets:
                    x1, y1, x2, y2 = map(int, det[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)
                    curr_roi_rect = (x1, y1, x2, y2)
                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid ROI: ({x1},{y1})-({x2},{y2})")
                        continue
                    # 分割+找点
                    roi = frame[y1:y2, x1:x2]
                    roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                    masks, _, _ = model_cellpose.eval(roi_gray, diameter=None)
                    contours, _ = cv.findContours(masks.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    print(f"ROI({x1},{y1}) segmented into {len(contours)} contours")
                    target_points, dyn_key, dyn_end, dyn_mid, dyn_diff = find_four_target_points(
                        contours, roi.shape[:2], frame_count, x1, y1,
                        step=step,
                        concave_angle_thresh=concave_angle_thresh
                    )
                    curr_dynamic_key = dyn_key
                    curr_dynamic_end = dyn_end
                    target_frame = [(x + x1, y + y1) for (x, y) in target_points]
                    curr_roi_info.append((curr_roi_rect, target_frame, dyn_key, dyn_end))
                    # 绘制轮廓+关键点
                    for cnt in contours:
                        cnt_frame = cnt + np.array([[x1, y1]], dtype=np.int32)
                        cv.drawContours(frame_copy, [cnt_frame], -1, (0, 255, 0), 2)

                    if len(target_frame) >= 2:
                        cv.circle(frame_copy, target_frame[0], 10, (0, 0, 255), -1)
                        cv.circle(frame_copy, target_frame[1], 10, (0, 0, 255), -1)
                        cv.putText(frame_copy, "L", (target_frame[0][0] + 12, target_frame[0][1] + 12),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                        cv.putText(frame_copy, "R", (target_frame[1][0] + 12, target_frame[1][1] + 12),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

                        if len(target_frame) == 4:
                            cv.circle(frame_copy, target_frame[2], 10, (255, 0, 0), -1)
                            cv.circle(frame_copy, target_frame[3], 10, (255, 0, 0), -1)
                            cv.putText(frame_copy, "F1", (target_frame[2][0] + 12, target_frame[2][1] + 12),
                                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                            cv.putText(frame_copy, "F2", (target_frame[3][0] + 12, target_frame[3][1] + 12),
                                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

            # 相邻帧对比
            if len(curr_roi_info) > 0 and len(frame_cache) > 0:
                prev_frame_data = frame_cache[-1]
                prev_frame_num, prev_roi_rect, prev_points, prev_is_key, _, prev_dyn_key, prev_dyn_end = prev_frame_data

                for curr_roi, curr_points, curr_key, curr_end in curr_roi_info:
                    iou = calculate_iou(curr_roi, prev_roi_rect)
                    if iou >= iou_match_thresh:
                        roi_match_status = f"Match_Success(IOU={iou:.2f})"
                        max_dist, avg_dist, valid_count = calculate_points_change(curr_points, prev_points)
                        endpoint_max_dist, endpoint_avg_dist = calculate_endpoint_change(curr_points, prev_points)
                        current_max_dist = max_dist
                        current_avg_dist = avg_dist
                        current_endpoint_max_dist = endpoint_max_dist
                        current_endpoint_avg_dist = endpoint_avg_dist

                        if (max_dist > curr_dynamic_key and
                                endpoint_max_dist > curr_dynamic_end and
                                frame_count not in key_frame_set):
                            is_key_frame = True
                            key_frame_set.add(frame_count)
                            compare_type = "Adjacent"
                            print(f"[Frame {frame_count}] 动态阈值触发关键帧！")
                    else:
                        roi_match_status = f"Match_Failed(IOU={iou:.2f})"

            elif len(curr_roi_info) > 0 and len(frame_cache) == 0:
                roi_match_status = "First_Frame(No_History)"
            # 跨帧回溯
            curr_has_full_points = any(len(points) == 4 for _, points, _, _ in curr_roi_info)
            if curr_has_full_points and len(frame_cache) > 0:
                print(f"\n[Frame {frame_count}] 触发跨帧回溯对比")
                for curr_roi, curr_points, curr_key, curr_end in curr_roi_info:
                    valid_history_frames = []
                    for hf in frame_cache:
                        h_num, h_roi, h_points, h_is_key, h_path, h_dkey, h_dend = hf
                        if len(h_points) == 4 and h_num not in key_frame_set:
                            gap = frame_count - h_num
                            valid_history_frames.append((gap, hf))

                    if not valid_history_frames:
                        continue
                    valid_history_frames.sort(key=lambda x: x[0])
                    min_gap, best_hf = valid_history_frames[0]
                    h_num, h_roi, h_points, h_is_key, h_path, h_dkey, h_dend = best_hf
                    iou = calculate_iou(curr_roi, h_roi)
                    if iou < iou_match_thresh:
                        continue
                    cross_max, cross_avg, _ = calculate_points_change(curr_points, h_points)
                    cross_end_max, _ = calculate_endpoint_change(curr_points, h_points)

                    if cross_max > curr_dynamic_key and cross_end_max > curr_dynamic_end:
                        key_frame_set.add(h_num)
                        h_frame = cv.imread(h_path)
                        if h_frame is not None:
                            h_key_frame = draw_key_frame_label(h_frame.copy(), cross_max, cross_end_max,
                                                               curr_dynamic_key, curr_dynamic_end)
                            cv.imwrite(os.path.join(key_frame_dir, f"key_frame_{h_num}_backtrack.jpg"), h_key_frame)

            # 绘制信息
            cv.rectangle(frame_copy, (orig_w - 320, 10), (orig_w - 10, 90), (0, 0, 0), -1)
            alpha = 0.7
            frame_copy[10:90, orig_w - 320:orig_w - 10] = cv.addWeighted(
                frame_copy[10:90, orig_w - 320:orig_w - 10], alpha,
                np.zeros_like(frame_copy[10:90, orig_w - 320:orig_w - 10]), 1 - alpha, 0)
            cv.putText(frame_copy, f"Max Dist: {current_max_dist:.2f}px", (orig_w - 310, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv.putText(frame_copy, f"Max Dist: {current_max_dist:.2f}px", (orig_w - 310, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(frame_copy, f"Endpoint Dist: {current_endpoint_max_dist:.2f}px", (orig_w - 310, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv.putText(frame_copy, f"Endpoint Dist: {current_endpoint_max_dist:.2f}px", (orig_w - 310, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv.putText(frame_copy, f"Dynamic Thresh: Key={curr_dynamic_key:.1f} End={curr_dynamic_end:.1f}",
                       (orig_w - 310, 85),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # 保存+日志
            save_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
            cv.imwrite(save_path, frame_copy)
            if is_key_frame and save_key_frames_separately:
                cv.imwrite(os.path.join(key_frame_dir, f"key_frame_{frame_count}.jpg"), frame_copy)

            with open(log_path, "a", encoding="utf-8") as f:
                tf = curr_roi_info[0][1] if len(curr_roi_info) > 0 else []
                e1x = tf[0][0] if len(tf) >= 1 else ""
                e1y = tf[0][1] if len(tf) >= 1 else ""
                e2x = tf[1][0] if len(tf) >= 2 else ""
                e2y = tf[1][1] if len(tf) >= 2 else ""
                f1x = tf[2][0] if len(tf) >= 3 else ""
                f1y = tf[2][1] if len(tf) >= 3 else ""
                f2x = tf[3][0] if len(tf) >= 4 else ""
                f2y = tf[3][1] if len(tf) >= 4 else ""
                f.write(f"{frame_count},{e1x},{e1y},{e2x},{e2y},{f1x},{f1y},{f2x},{f2y},{1 if is_key_frame else 0},"
                        f"{current_max_dist:.2f},{current_avg_dist:.2f},{current_endpoint_max_dist:.2f},{current_endpoint_avg_dist:.2f},"
                        f"{compare_type},{roi_match_status},{curr_dynamic_key:.2f},{curr_dynamic_end:.2f}\n")

            if len(curr_roi_info) > 0 and len(curr_roi_info[0][1]) >= 2:
                item = (frame_count, curr_roi_info[0][0], curr_roi_info[0][1], is_key_frame, orig_frame_path,
                        curr_roi_info[0][2], curr_roi_info[0][3])
                frame_cache.append(item)
                if len(frame_cache) > max_backtrack_frames:
                    frame_cache.pop(0)
            frame_time = (time.perf_counter() - start_time) * 1000
            total_time += frame_time
            print(f"[Frame {frame_count}] 耗时{frame_time:.1f}ms | 动态Key阈值={curr_dynamic_key:.1f}")
            frame_count += 1
    cap.release()
    # 最初的波动帧为关键帧
    if save_key_frames_separately and os.path.exists(key_frame_dir):
        key_files = [f for f in os.listdir(key_frame_dir) if f.endswith(('.jpg', '.png'))]
        frame_numbers = []
        file_map = {}
        for f in key_files:
            try:
                num_str = ''.join([c for c in f if c.isdigit()])
                frame_num = int(num_str)
                frame_numbers.append(frame_num)
                file_map[frame_num] = os.path.join(key_frame_dir, f)
            except:
                continue
        if frame_numbers:
            min_frame = min(frame_numbers)
            keep_file = file_map[min_frame]
            for fn, fp in file_map.items():
                if fn != min_frame:
                    os.remove(fp)
    print(f"\n处理完成！总帧数：{frame_count} | 关键帧总数：{len(key_frame_set)}")

if __name__ == "__main__":
    video_path ="example/demo.avi"
    model_path = "weights/yolov5_best.pt"
    main_process(
        video_path=video_path,
        model_path=model_path,
        save_dir="outputs",
        step=1,
        concave_angle_thresh=180,
        save_key_frames_separately=True,
        iou_match_thresh=0.3,
        max_backtrack_frames=2
    )