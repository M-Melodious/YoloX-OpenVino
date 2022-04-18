import cv2
import numpy as np

from filevideostream import VideoCap
from yolox.preprocess import preproc_buffer
from yolox.visualize import draw_bbox, concat_tile
from yolox.utils import multiclass_nms, demo_postprocess


def get_video_caps(video_list:list, resize:tuple=None,
                   buf_size:int=64) -> list:
    """
    Helper function to create a VideoCap thread with processing
    for every video source.

    Parameters
    ----------
        video_list: list
            list of video source paths
        resize: tuple
            resize tuple for preprocessing, default: (640, 640)
            Note: default value is based on the network used.
        buf_size: int
            buffer size to store the decoded frames in background
            Default: 64

    Returns
    -------
        video_caps list
    
    """
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


def get_video_writer(out_name:str, fps:int=30, size:tuple=None):
    """
    Helper function to create video writer.
    
    """
    assert size, "Size needed for video writer"
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    return cv2.VideoWriter(out_name, fourcc, fps, size)


def stop_video_caps(vid_caps:list):
    """
    Helper function to stop the video cap thread in background
    
    """
    for vid in vid_caps:
        vid.fvs.stop()


def batchify_frames(frames:list) -> np.ndarray:
    """
    Helper function to batchify the given list of frames

    Parameters
    ----------
        frames: list
            list of preprocessed frames

    Returns
    -------
        np array
        
    """
    return np.array(frames)


def get_detections(video_caps:list, output:np.ndarray,
                   net_shape:tuple=(640,640), nms_thresh:float=0.45,
                   score_thresh:float=0.1) -> None:
    """
    Helper function to post process the detections.

    Parameters
    ----------
        video_caps: list
            list of video caps objects
        output: np.ndarray
            output from inference net
        net_shape: tuple
            preprocess shape, default: (640, 640)
            Note: it is based on the net used
        nms_thresh: float
            nms threshold, default: 0.45
        score_thresh: float
            score thresh, default: 0.1

    """
    predictions = demo_postprocess(output, net_shape)

    ## Loop through batch predictions and get detections
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


def put_highlighted_text(frame:np.ndarray, message:str, position:tuple,
                         font_face:int, font_scale:float, color:tuple,
                         thickness:int) -> None:
    """
    Helper function to put the text on the image.
    
    """
##    cv2.putText(frame, message, position, font_face, font_scale,
##                (255, 255, 255), thickness + 1) # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)
    

def visualize_multicam_detections(frames:list, video_caps:list,
                                  max_window_size:tuple=(1280, 1040),
                                  score_thr:float=0.1) -> np.ndarray:
    """
    Helper function to draw bboxes over images and make grid out of
    image batch.

    Parameters
    ----------
        frames: list
            list of frames
        video_caps: list
            list of video caps object
        max_window_size: tuple
            max size for grid, default: (1280, 1040)
        score_thr: float
            score threshold, default: 0.1

    Returns
    -------
        grid image
        
    """
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
