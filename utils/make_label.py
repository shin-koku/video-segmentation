from collections import OrderedDict
import os
import sys

def read_annotation(train_split, test_split):
    train_set = OrderedDict()
    test_set = OrderedDict()
    if train_split is not None:
        with open(train_split) as train:
            for line in train.readline():
                info_list = line.split()
                action_id = info_list[1]
                data_list = info_list[0].split('-')
                frame_start = int(data_list[-2][1:])
                frame_end = int(data_list[-1][1:])
                video_name = '-'.join([data_list[0], data_list[1], data_list[2]])
                if video_name not in train_set:
                    set = {}
                    action_list = []
                    interval = [frame_start, frame_end, action_id]
                    action_list.append(interval)
                    set[video_name] = interval
                    train_set.update(set)
                else:
                    interval = [frame_start, frame_end, action_id]
                    train_set[video_name].append(interval)
            train.close()
    if test_split is not None:
        with open(test_split) as test:
            for line in test.readline():
                info_list = line.split()
                action_id = info_list[1]
                data_list = info_list[0].split('-')
                frame_start = int(data_list[-2][1:])
                frame_end = int(data_list[-1][1:])
                video_name = '-'.join([data_list[0], data_list[1], data_list[2]])
                if video_name not in test_set:
                    set = {}
                    action_list = []
                    interval = [frame_start, frame_end, action_id]
                    action_list.append(interval)
                    set[video_name] = interval
                    test_set.update(set)
                else:
                    interval = [frame_start, frame_end, action_id]
                    test_set[video_name].append(interval)
            test.close()
    return train_set,test_set

def make_frame_label(video_label, n_frames_pth):
    frames_num = int(open(n_frames_pth).read())
    ##after python 3.6 normal has maintained order of insertion, but still as regard whose ptthon version still not reach that
    ##use Ordereddict for safety
    label_dict = OrderedDict({idx: 0} for idx in range(1, frames_num + 1))
    for label in video_label:
        frame_start = label[0]
        frame_end = label[1]
        action = label[3]
        label_dict.update({index: action} for index in range(frame_start, frame_end + 1))

    return label_dict


def write_label(label_dict, video_name, output_pth):
    write_file = open(os.path.join(output_pth, video_name + ".txt"), 'w')
    for k, v in label_dict:
        line = write_file.write(k+","+v+"\n")

    write_file.close()

od = OrderedDict({1:1,2:2,3:3})
print(len(od))

if __name__ == "__main__":
    train_pth = sys.argv[1]
    test_pth = sys.argv[2]
    output_pth = sys.argv[3]
    n_frames_pth = sys.argv[4]
    train_split,test_split = read_annotation(train_pth,test_pth)
    if len(train_split)>0:
        for video_name,video_label in train_split:
            label_dict = make_frame_label(video_label,os.path.join(n_frames_pth,video_name))
            write_label(label_dict,video_name,output_pth)