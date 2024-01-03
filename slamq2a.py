import numpy as np
import gtsam
import matplotlib.pyplot as plt

def read_g2o_file_3d(input_INTEL_g2o):
    poses_dict = {}
    edges_dict = {}

    with open(input_INTEL_g2o, 'r') as f:
        for line in f:
            elements = line.split()
            if elements[0] == 'VERTEX_SE3:QUAT':
                key = elements[1]
                values = np.array([float(x) for x in elements[2:]])
                poses_dict[key] = values

            elif elements[0] == 'EDGE_SE3:QUAT':
                key = (elements[1], elements[2])
                info = np.array([float(x) for x in elements[10:]])
                information_matrix = np.array([[info[0], info[1], info[2], info[3], info[4], info[5]],
                                         [info[1], info[6], info[7], info[8], info[9], info[10]],
                                         [info[2], info[7], info[11], info[12], info[13], info[14]],
                                         [info[3], info[8], info[12], info[15], info[16], info[17]],
                                         [info[4], info[9], info[13], info[16], info[18], info[19]],
                                         [info[5], info[10], info[14], info[17], info[19], info[20]]])
                cov_matrix = np.linalg.inv(information_matrix)
                values = np.array([elements[3], elements[4], elements[5], elements[6], elements[7],elements[8], elements[9], cov_matrix])
                edges_dict[key] = values

    print(len(poses_dict)) # output 1228
    print(len(edges_dict)) # output 1483
    return poses_dict, edges_dict

if __name__ == "__main__":
    parking_garage = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/parking-garage.g2o"
    x, y = read_g2o_file_3d(parking_garage)
    print(y)
