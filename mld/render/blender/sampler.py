import numpy as np

def get_frameidx(*, mode, nframes, exact_frame, frames_to_keep):
    if mode == "sequence":
        # [0, 32, 65, 98, 130, 162, 195]
        # frames_to_keep = 10
        frames_to_keep = 3
        frameidx = np.linspace(0, nframes - 1, frames_to_keep)
        frameidx = np.round(frameidx).astype(int)
        frameidx = list(frameidx)

        
        # frameidx = [0,1] #for content comparison
        print("nframes",nframes)


        # frameidx = [ 49, 195]
    elif mode == "frame":
        index_frame = int(exact_frame*nframes)
        frameidx = [index_frame]
    elif mode == "video":
        frameidx = range(0, nframes)
    else:
        raise ValueError(f"Not support {mode} render mode")
    return frameidx
