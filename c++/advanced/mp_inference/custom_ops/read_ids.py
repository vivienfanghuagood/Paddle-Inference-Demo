import paddle
import time
import os
import numpy as np
import struct


def deserialize_from_file(fp):
    # dtype
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b'0':
        data = fp.read(4)
        data_out = struct.unpack("f",data)[0]
        while data:
            data_out = struct.unpack("f",data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b'1':
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l",data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b'2':
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i",data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


if __name__ == "__main__":
    try:
        target = "../real_time_save.temp_ids_rank_0_step_0"
        if os.path.getsize(target) > 0 :
            while True:
                fp = open(target,"rb+")
                flg = fp.read(1)
                if flg == b'1':
                    break
                else:
                    print("waiting")     
            data_list = deserialize_from_file(fp)
            print(data_list.shape)
            print(data_list)          
    except EOFError:
        print("eof error")

    
