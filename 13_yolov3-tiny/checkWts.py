import os

with open("./yolov3_tiny.wts", "r") as fp:
    lines = [line.strip() for line in fp]

with open("./wts_name.txt", "w") as fp:
    for line in lines:
        res = line.split(" ")
        # print(res[0])
        fp.write(res[0])
        fp.write("\n")