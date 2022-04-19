import cv2
import sys
import time
import logging
import threading

from enum import Enum
from collections import deque
from time import perf_counter

from config import post_process, visualization

from helpers import (
    stop_video_caps,
    batchify_frames,
    get_detections,
    put_highlighted_text,
    visualize_multicam_detections
)


logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class Mode():
    def __init__(self, value):
        self.current = value

    def next(self):
        if self.current.value + 1 < len(Modes):
            self.current = Modes(self.current.value + 1)
        else:
            self.current = Modes(0)


class ModeInfo():
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def async_callback(status, callback_args):
    (request, frame_id, frame_mode, frame,
     start_time, completed_request_results, empty_requests,
     mode, event, callback_exceptions) = callback_args

    try:
        if status != 0:
            raise RuntimeError(f'Infer Request has returned status code {status}')

        completed_request_results[frame_id] = (frame, request.output_blobs,
                                               start_time, frame_mode == mode.current)

        if mode.current == frame_mode:
            empty_requests.append(request)
    except Exception as e:
        callback_exceptions.append(e)

    event.set()

def await_requests_completion(requests):
    for request in requests:
        request.wait()


def run_async_n(infer_net, threaded_gen, video_caps,
                video_writer, net_shape=(640,640)):
    """
    Function for n async inference.
    
    
    """
    mode = Mode(Modes.USER_SPECIFIED)
    mode_info = { mode.current: ModeInfo() }

    empty_requests = deque(infer_net.net_plugin.requests)
    completed_request_results = {}
    next_batch_id = 0
    next_batch_id_to_show = 0
    event = threading.Event()
    callback_exceptions = []

    while (threaded_gen.running
           or completed_request_results
           or len(empty_requests) < len(infer_net.net_plugin.requests)) \
           and not callback_exceptions:
        if next_batch_id_to_show in completed_request_results:
            frames, output, start_time, is_same_mode = \
                    completed_request_results.pop(next_batch_id_to_show)

            next_batch_id_to_show += 1

            if is_same_mode:
                mode_info[mode.current].frames_count += 1

            res = output[infer_net.out_blob].buffer
            get_detections(video_caps, res, net_shape=net_shape,
                           nms_thresh=post_process.nms_thr,
                           score_thresh=post_process.score_thr)

            vis = visualize_multicam_detections(frames, video_caps,
                                                max_window_size=visualization.max_window_size,
                                                score_thr=post_process.score_thr)

            if mode_info[mode.current].frames_count != 0:
                fps_message = "FPS: {:.2f}".format(
                    mode_info[mode.current].frames_count / \
                    (perf_counter() - mode_info[mode.current].last_start_time))
                mode_info[mode.current].latency_sum += perf_counter() - start_time
                latency_message = "Latency: {:.1f} ms".format(
                    (mode_info[mode.current].latency_sum / \
                     mode_info[mode.current].frames_count) * 1e3)

                put_highlighted_text(vis, fps_message, (15, 20),
                                     cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                put_highlighted_text(vis, latency_message, (15, 50),
                                     cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

            video_writer.write(vis)
            
        elif empty_requests and threaded_gen.running:
            start_time = perf_counter()
            frames = next(threaded_gen, None)

            if not frames:
                stop_video_caps(video_caps)
                continue

            proc_frames = batchify_frames([f[1] for f in frames])
            frames = [f[0] for f in frames]

            request = empty_requests.popleft()

            # Start inference
            request.set_completion_callback(py_callback=async_callback,
                                            py_data=(request,
                                                     next_batch_id,
                                                     mode.current,
                                                     frames,
                                                     start_time,
                                                     completed_request_results,
                                                     empty_requests,
                                                     mode,
                                                     event,
                                                     callback_exceptions))

            request.async_infer(inputs={input_blob: proc_frames})
            next_batch_id += 1

        else:
            event.wait()

    if callback_exceptions:
        raise callback_exceptions[0]

    for mode_value in mode_info.keys():
        log.info("")
        log.info("Mode: {}".format(mode_value.name))

        end_time = mode_info[mode_value].last_end_time if mode_value in mode_info \
                                                          and mode_info[mode_value].last_end_time is not None \
                                                       else perf_counter()
        log.info("FPS: {:.1f}".format(mode_info[mode_value].frames_count / \
                                      (end_time - mode_info[mode_value].last_start_time)))
        log.info("Latency: {:.1f} ms".format((mode_info[mode_value].latency_sum / \
                                             mode_info[mode_value].frames_count) * 1e3))

    await_requests_completion(infer_net.net_plugin.requests)

    return
