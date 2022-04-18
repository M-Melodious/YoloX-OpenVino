import os
import cv2
import sys
import time
import argparse
import numpy as np
import logging as log

from pathlib import Path

from yolox.utils import mkdir

from inference import Network
from filevideostream import VideoCap, ThreadedGenerator
from config import inference, vid_sources, post_process, visualization

from helpers import (
    get_video_caps,
    get_video_writer,
    stop_video_caps,
    batchify_frames,
    get_detections,
    put_highlighted_text,
    visualize_multicam_detections
)
    

def _run_inference_async(infer_net, threaded_gen, video_caps, video_writer):
    """
    Function for async inference.
    
    """
    cur_batch_id = 0
    next_batch_id = 1
    
    _, c, h, w = infer_net.get_input_shape()
    start_time = time.time()

    ## Get the fist batch and perform inference
    prev_frames = next(threaded_gen)
    infer_net.exec_net_async(cur_batch_id,
                             batchify_frames([frame[1] for frame in prev_frames]))

    ## Loop through the batches and inference in async mode
    for frames in threaded_gen:
        
        images = batchify_frames([frame[1] for frame in frames])
        infer_net.exec_net_async(next_batch_id, images)

        ## If previous inference is done grab the output and post process it
        if infer_net.wait(cur_batch_id) == 0:
            res = infer_net.get_output_async(cur_batch_id)
            get_detections(video_caps, res, net_shape=(h, w),
                           nms_thresh=post_process.nms_thr,
                           score_thresh=post_process.score_thr)

        ## Swap batch ids
        cur_batch_id, next_batch_id = next_batch_id, cur_batch_id

        ## Draw bboxes and make grid out of the batch
        vis = visualize_multicam_detections(prev_frames, video_caps,
                                            max_window_size=visualization.max_window_size,
                                            score_thr=post_process.score_thr)

        ## Statistics
        fps_time = time.time() - start_time
        fps_message = f"FPS: {1/fps_time:.3f}"
        inf_time_message = f"Inference time: N/A for async mode"
        put_highlighted_text(vis, fps_message, (15, 30),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
        put_highlighted_text(vis, inf_time_message, (15, 60),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

        ## Write frame to output video
        video_writer.write(vis)

        prev_frames, frames = frames, prev_frames

    ## Stop the video threads
    stop_video_caps(video_caps)
    

def _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer):
    """
    Function for sync inference.
    
    """
    _, c, h, w = infer_net.get_input_shape()
    det_time = 0
    
    start_time = time.time()

    ## Loop through batches and perform inference
    for frames in threaded_gen:

        det_start = time.time()
        images = batchify_frames([frame[1] for frame in frames])
        infer_net.exec_net(images)
        det_time = time.time() - det_start

        ## Get output and post process it
        res = infer_net.get_output()
        get_detections(video_caps, res, net_shape=(h, w),
                       nms_thresh=post_process.nms_thr,
                       score_thresh=post_process.score_thr)

        ## Draw bboxes and make grid out of the batch
        vis = visualize_multicam_detections(frames, video_caps,
                                            max_window_size=visualization.max_window_size,
                                            score_thr=post_process.score_thr)

        ## Statistics
        fps_time = time.time() - start_time
        fps_message = f"FPS: {1/fps_time:.3f}"
        inf_time_message = f"Inference time: {det_time*1000:.3f}, Sync Mode"
        put_highlighted_text(vis, fps_message, (15, 30),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
        put_highlighted_text(vis, inf_time_message, (15, 60),
                             cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)


        ## Write grid frame to output video
        cv2.imwrite("temp.jpg", vis)
##        video_writer.write(vis)

    ## Stop all video threads
    stop_video_caps(video_caps)

        

def main():
    """
    Main entry point function.
    
    """
    ## Create network and load the weights
    infer_net = Network()
    infer_net.load_model(inference.model, inference.device,
                         num_requests=inference.num_requests,
                         batch_size=inference.batch_size)
    _, c, h, w = infer_net.get_input_shape()

    ## Create video threads and start the threads to fill up the buffer in background
    videos_list = list(Path(vid_sources.source_dir).glob("*.*"))
    video_caps = get_video_caps([vid.as_posix() for vid in videos_list],
                                resize=(h,w), buf_size=32)

    threaded_gen = ThreadedGenerator(video_caps).__iter__()

    ## Create video writer
    mkdir(visualization.output_dir)
    vid_name = f"{len(video_caps)}-channel-{inference.mode}-yolox-x.avi"
    out_video_name = Path(visualization.output_dir, vid_name).as_posix()
    video_writer = get_video_writer(out_video_name, size=visualization.target_size)

    ## Perform inference according to mode specify in config file
    if inference.mode == 'async':
        print(f"Inference in async mode")
        _run_inference_async(infer_net, threaded_gen, video_caps, video_writer)
    elif inference.mode == 'sync':
        print(f"Inference in sync mode")
        _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer)
    else:
        print(f"Unknown mode!!! Please either use async or sync")

    ## Release the object
    video_writer.release()
    print("Done!!!")



            





if __name__ == '__main__':
    sys.exit(main() or 0)
