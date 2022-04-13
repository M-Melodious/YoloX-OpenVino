# import the necessary packages
import sys
import cv2
import time
import datetime
import numpy as np

from queue import Queue
from threading import Thread


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class FileVideoStream:
    def __init__(self, path, transform=None,
                 transform_args={}, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform
        self.transform_args = transform_args

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while not self.stopped:
##            # if the thread indicator variable is set, stop the
##            # thread
##            if self.stopped:
##                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True
                    break
                        
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame, **self.transform_args)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()



class VideoCap:
    """
    Store the data of video

    """
    def __init__(self, vid_file, preproc=None,
                 preproc_args=None, buf_size=64):
        self.fvs = FileVideoStream(vid_file,
                                   transform=preproc,
                                   transform_args=preproc_args,
                                   queue_size=buf_size).start()
        
        self.height, self.width = None, None
        self.ratio = None
        self.frame = None
        self.dets = None
        self.blank_frame = None
        self.blank_proc_frame = None
        self._get_video_dimension()

        if preproc and preproc_args:
            self._get_preproc_ratio(input_size=preproc_args['input_size'])
            self.blank_frame, self.blank_proc_frame = preproc(
                np.zeros((self.height, self.width, 3)), **preproc_args)

    def _get_video_dimension(self):
        self.width = int(round(self.fvs.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = int(round(self.fvs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _get_preproc_ratio(self, input_size=(640,640)):
        self.ratio = min(input_size[0] / self.height, input_size[1] / self.width)

    def read(self):
        frames = self.fvs.read()

        return frames if frames else (self.blank_frame, self.blank_proc_frame)


class FramesThreadBody:
    def __init__(self, vid_caps, max_queue_length=128):
        self.process = True
        self.frames_queue = Queue()
        self.proc_frames_queue = Queue()
        self.vid_caps = vid_caps
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue

            if not all([vid.fvs.running() for vid in self.vid_caps]):
                self.process = False
                break

            frames = self._get_frames()
            self.frames_queue.put([frame[0] for frame in frames])
            self.proc_frames_queue.put([frame[1] for frame in frames])

    def _get_frames(self):
        return [vid.read() for vid in self.vid_caps]


class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """

    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=128,
                 daemon=True,
                 Thread=Thread,
                 Queue=Queue):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
##            for value in self._iterator:
##                self._queue.put(value)
            while True:
                if not self._queue.full():
                    self._queue.put([vid.read() for vid in self._iterator])

                    if not all([vid.fvs.running() for vid in self._iterator]):
                        break
                else:
                    time.sleep(0.1)

        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value

        self._thread.join()



def get_frames(vid_caps):
    while True:
        yield [vid.read() for vid in vid_caps]

        if not all([vid.fvs.running() for vid in vid_caps]):
            for vid in vid_caps:
                vid.fvs.stop()
            break
