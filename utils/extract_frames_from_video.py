import os
import subprocess

def read_file(video_path,output_path):
    if not os.path.exists(video_path) or not os.path.exists(output_path):
        raise NotADirectoryError("check the video directory or output directory whether exist or not")

    video_list = os.listdir(video_path)
    if video_list == None:
        raise FileNotFoundError

    for r in video_list:
        if '.mp4' not in r:
            continue
        file_name = r.split('.')[0]

        dst_path = os.path.join(video_path,file_name)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        else:
            os.rmdir(dst_path)
            os.mkdir(dst_path)
        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(os.path.join(video_path,r),dst_path)
        subprocess.call(cmd,shell=True)



read_file("/home/shinkoku/materials/GTEA gaze video clips/raw_video","/home/shinkoku/materials/GTEA gaze video clips/raw_video")




