import subprocess as sp
import shlex


# render video for submission
def render_video(frames, w, h, path_video):
    target_fps= = 30
    vid = sp.Popen(shlex.split(f'ffmpeg -y -s {w}x{h} -pixel_format bgr24 -f rawvideo -r {target_fps} -i pipe: -vcodec h264 -pix_fmt yuv420p -crf 18 "{path_video}"'), stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    for frame in frames:
        vid.stdin.write(frame.tobytes())
    vid.stdin.close()
    vid.wait()
    vid.terminate()


if __name__ == '__main__':
    # load model

    # read 3000 videos from data/test/unsupervised
    render_video(frames, w, h, path_video)
