import os
import cv2
import sys
import time
import argparse
import numpy as np
import logging as log

from pathlib import Path

from yolox.utils import mkdir
from run_async_n import run_async_n

from inference import Network
from filevideostream import FPS, VideoCap, ThreadedGenerator
from config import inference, vid_source, post_process, visualization

from helpers import (
    get_video_caps,
    get_video_writer,
    stop_video_caps,
    batchify_frames,
    get_detections,
    put_highlighted_text,
    visualize_multicam_detections
)


## Logger
log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
logger = log.getLogger()



def _run_inference_async(infer_net, threaded_gen, video_caps, video_writer):
    """
    Function for async inference.
    
    """
    cur_batch_id = 0
    next_batch_id = 1
    
    _, c, h, w = infer_net.get_input_shape()
    start_time = time.time()

    fps = FPS().start()
    
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

        ## Update fps stats
        fps.update()

    ## Stop the video threads
    stop_video_caps(video_caps)

    ## Stop fps and log the stats
    fps.stop()
    logger.info(f"Elapsed time: {fps.elapsed():.2f} seconds")
    logger.info(f"Approx. FPS: {fps.fps():.2f}")

    return
    

def _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer):
    """
    Function for sync inference.
    
    """
    _, c, h, w = infer_net.get_input_shape()
    det_time = 0
    
    start_time = time.time()
    fps = FPS().start()

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
        video_writer.write(vis)

        ## Update fps stats
        fps.update()

    ## Stop all video threads
    stop_video_caps(video_caps)

    ## Stop fps and log the stats
    fps.stop()
    logger.info(f"Elapsed time: {fps.elapsed():.2f} seconds")
    logger.info(f"Approx. FPS: {fps.fps():.2f}")

    return

        

def main():
    """
    Main entry point function.
    
    """
    ## Create network and load the weights
    logger.info("Loading weights...")
    infer_net = Network()
    infer_net.load_model(inference.model, inference.device,
                         num_requests=inference.num_requests,
                         batch_size=inference.batch_size)
    _, c, h, w = infer_net.get_input_shape()

    logger.info("Creating background video threads...")
    ## Create video threads and start the threads to fill up the buffer in background
    videos_list = list(Path(vid_source.source_dir).glob("*.*"))
    video_caps = get_video_caps([vid.as_posix() for vid in videos_list],
                                resize=(h,w), buf_size=vid_source.buf_size)

    logger.info("Starting threaded generator in background...")
    threaded_gen = ThreadedGenerator(video_caps).__iter__()

    ## Create video writer
    logger.info("Getting video writer (XVID -> AVI)...")
    mkdir(visualization.output_dir)
    vid_name = f"{len(video_caps)}-channel-{inference.mode}.avi"
    out_video_name = Path(visualization.output_dir, vid_name).as_posix()
    video_writer = get_video_writer(out_video_name, codec='XVID',
                                    size=visualization.target_size)

    ## Perform inference according to mode specify in config file
    if inference.mode == 'async':
        logger.info("Running inference in async mode...")
##        _run_inference_async(infer_net, threaded_gen, video_caps, video_writer)
        run_async_n(infer_net, threaded_gen, video_caps,
                    video_writer, net_shape=(h,w))
    elif inference.mode == 'sync':
        logger.info("Running inference in sync mode...")
        _run_inference_sync(infer_net, threaded_gen, video_caps, video_writer)
    else:
        logger.error("Unknown inference mode!!!")

    ## Release the object
    video_writer.release()
    logger.info(f"Inference video saved to {out_video_name}...")
    logger.info("Inference done!")



            





if __name__ == '__main__':
    sys.exit(main() or 0)

