import numpy as np
# from gtsam import Pose2, Rot2, Point2, noiseModel, NonlinearFactorGraph, Values
import gtsam
import matplotlib.pyplot as plt

def read_g2o_file_2d(input_INTEL_g2o):
    poses_dict = {}
    edges_dict = {}

    with open(input_INTEL_g2o, 'r') as f:
        for line in f:
            elements = line.split()
            if elements[0] == 'VERTEX_SE2':
                # elements = elements.split()
                key = elements[1]
                values = np.array([float(x) for x in elements[2:]])
                poses_dict[key] = values

            elif elements[0] == 'EDGE_SE2':
                key = (elements[1], elements[2])
                info_matrix = np.zeros((3,3))
                cov_matrix = np.zeros((3,3))
                edge_info = np.array([float(x) for x in elements[6:]]) 
                info_matrix[0, 0] = edge_info[0]
                info_matrix[0, 1] = info_matrix[1, 0] = edge_info[1]
                info_matrix[0, 2] = info_matrix[2, 0] = edge_info[2]
                info_matrix[1, 1] = edge_info[3]
                info_matrix[1, 2] = info_matrix[2, 1] = edge_info[4]
                info_matrix[2, 2] = edge_info[5]
                cov_matrix = np.linalg.inv(info_matrix)
                values = np.array([elements[3], elements[4], elements[5], cov_matrix])
                edges_dict[key] = values

    # print(len(poses_dict)) # output 1228
    print(len(edges_dict)) # output 1483
    return poses_dict, edges_dict

if __name__ == "__main__":
    input_INTEL_g2o = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/input_INTEL_g2o.g2o"
    x, y = read_g2o_file_2d(input_INTEL_g2o)
    print(y)
    # print(len(x), len(y))


