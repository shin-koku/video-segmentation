import os


def generate_n_frames(frames_path):
    file_list = os.listdir(frames_path)
    file_list = filter(lambda path:"." not in path ,file_list)
    for frames_file in file_list:
        pth = os.path.join(frames_path,frames_file)
        n_frames_txt = open(pth+"/n_frames.txt",'w')
        n_frames = str(len(list(filter(lambda x : ".jpg" in x ,os.listdir(pth)))))
        n_frames_txt.write(n_frames)
        n_frames_txt.close()


generate_n_frames("/home/shinkoku/materials/GTEA gaze video clips/raw_video")