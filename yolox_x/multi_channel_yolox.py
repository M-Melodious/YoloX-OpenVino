import os
import cv2
import sys
import time
import argparse
import numpy as np
import logging as log

from pathlib import Path

from yolox.visualize import draw_bbox, concat_tile
from yolox.preprocess import preproc_buffer
from yolox.coco_classes import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess

from inference import Network
from filevideostream import VideoCap, ThreadedGenerator
from config import inference, vid_sources, post_process, visualization



def get_video_caps(video_list, resize=None, buf_size=64):
    video_caps = []
    preproc_args = {'input_size': resize if resize else (640,640),
                    'swap': (2,0,1)
                    }

    for vid in video_list:
        video_cap = VideoCap(vid, preproc=preproc_buffer,
                             preproc_args=preproc_args,
                             buf_size=buf_size)

        video_caps.append(video_cap)

    return video_caps

def get_video_writer(out_name, fps=30, size=None):
    assert size, "Size needed for video writer"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(out_name, fourcc, fps, size)

def stop_video_caps(vid_caps):
    for vid in vid_caps:
        vid.fvs.stop()

def batchify_frames(frames):
    return np.array(frames)

def post_process(video_caps, output, net_shape=(640,640),
                 nms_thresh=0.45, score_thresh=0.1):
    predictions = demo_postprocess(output, net_shape)

    for i in range(len(predictions)):
        preds = predictions[i].copy()
        boxes = preds[:, :4]
        scores = preds[:, 4, None] * preds[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= video_caps[i].ratio

        video_caps[i].dets = multiclass_nms(boxes_xyxy, scores,
                                            nms_thr=nms_thresh,
                                            score_thr=score_thresh)

##def get_target_size(frame_sizes, vis=None,
##                    max_window_size=(1920, 1080), stack_frames='vertical'):
##    if vis is None:
##        width = 0
##        height = 0
##        
##        for size in frame_sizes:
##            if width > 0 and height > 0:
##                if stack_frames == 'vertical':
##                    height += size[1]
##                elif stack_frames == 'horizontal':
##                    width += size[0]
##            else:
##                width, height = size
##    else:
##        height, width = vis.shape[:2]
##
##    if stack_frames == 'vertical':
##        target_height = max_window_size[1]
##        target_ratio = target_height / height
##        target_width = int(width * target_ratio)
##        
##    elif stack_frames == 'horizontal':
##        target_width = max_window_size[0]
##        target_ratio = target_width / width
##        target_height = int(height * target_ratio)
##        
##    return target_width, target_height

def put_highlighted_text(frame, message, position,
                         font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale,
                (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)

def visualize_multicam_detections(frames, video_caps,
                                  max_window_size=(1280, 1040), score_thr=0.1):
    vis = None
    default_size = (640, 480)
    tile_frames = []
    for i, vid in enumerate(video_caps):
        if vid.dets is not None:
            final_boxes = vid.dets[:, :4]
            final_scores, final_cls_inds = vid.dets[:, 4], vid.dets[:, 5]
            draw_bbox(frames[i][0], final_boxes, final_scores,
                      final_cls_inds, conf=score_thr)
            tile_frames.append(cv2.resize(frames[i][0], default_size))

    grid = concat_tile([tile_frames[:2], tile_frames[2:]])
    if (grid.shape[0] > max_window_size[1]
        or grid.shape[1] > max_window_size[0]):
        grid = cv2.resize(grid, max_window_size)

    del tile_frames

    return grid
    

def _run_inference_async(infer_net, threaded_gen, video_caps, video_writer):
    cur_batch_id = 0
    next_batch_id = 1
    
    _, c, h, w = infer_net.get_input_shape()
    start_time = time.time()

    prev_frames = next(threaded_gen)
    infer_net.exec_net_async(cur_batch_id,
                             batchify_frames([frame[1] for frame in prev_frames]))

    for frames in threaded_gen:
        
        images = batchify_frames([frame[1] for frame in frames])
        infer_net.exec_net_async(next_batch_id, images)

        if infer_net.wait(cur_batch_id) == 0:
            res = infer_net.get_output_async(cur_batch_id)
            post_process(video_caps, res, net_shape=(h, w),
                         nms_thresh=post_process.nms_thr,
                         score_thresh=post_process.score_thr)

        cur_batch_id, next_batch_id = next_batch_id, cur_batch_id

        vis = visualize_multicam_detections(prev_frames, video_caps,
                                            max_window_size=visualization.max_window_size,
                                            score_thr=post_process.score_thr)

        fps_time = time.time() - start_time
        fps_message = f"FPS: {1/fps_time:.3f}"
        inf_time_message = f"Inference time: N/A for async mode"
        put_highlighted_text(vis, fps_message, (15, 30),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
        put_highlighted_text(vis, inf_time_message, (15, 60),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

        video_writer.write(vis)

        prev_frames, frames = frames, prev_frames

    stop_video_caps(video_caps)

def _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer):
    _, c, h, w = infer_net.get_input_shape()
    det_time = 0
    
    start_time = time.time()

    for frames in threaded_gen:

        det_start = time.time()
        images = batchify_frames([frame[1] for frame in frames])
        infer_net.exec_net(images)
        det_time = time.time() - det_start
        
        res = infer_net.get_output()
        post_process(video_caps, res, net_shape=(h, w),
                     nms_thresh=post_process.nms_thr,
                     score_thresh=post_process.score_thr)

        vis = visualize_multicam_detections(frames, video_caps,
                                            max_window_size=visualization.max_window_size,
                                            score_thr=post_process.score_thr)

        fps_time = time.time() - start_time
        fps_message = f"FPS: {1/fps_time:.3f}"
        inf_time_message = f"Inference time: {det_time*1000:.3f}, Sync Mode"
        put_highlighted_text(vis, fps_message, (15, 30),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
        put_highlighted_text(vis, inf_time_message, (15, 60),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)


        video_writer.write(vis)

    stop_video_caps(video_caps)

        

def main():

    infer_net = Network()
    infer_net.load_model(inference.model, inference.device,
                         num_requests=inference.num_requests,
                         batch_size=inference.batch_size)
    _, c, h, w = infer_net.get_input_shape()

    videos_list = list(Path(vid_sources.source_dir).glob("*.*"))
    video_caps = get_video_caps([vid.as_posix() for vid in videos_list],
                                resize=(h,w), buf_size=32)

    threaded_gen = ThreadedGenerator(video_caps).__iter__()

    mkdir(visualization.output_dir)
    out_video_name = Path(visualization.output_dir, "4-channel-yolox-x.mp4").as_posix()
    video_writer = get_video_writer(out_video_name, size=visualization.target_size)

    if inference.mode == 'async':
        _run_inference_async(infer_net, threaded_gen, video_caps, video_writer)
    elif inference.mode == 'sync':
        _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer)
    else:
        print(f"Unknown mode!!! Please either use async or sync")

    video_writer.release()
    print("Done!!!")



            





if __name__ == '__main__':
    sys.exit(main() or 0)
