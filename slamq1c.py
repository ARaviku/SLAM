import numpy as np
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
                values = np.array([float(poses) for poses in elements[2:]])
                poses_dict[key] = values

            elif elements[0] == 'EDGE_SE2':
                key = (elements[1], elements[2])
                info_matrix = np.zeros((3,3))
                cov_matrix = np.zeros((3,3))
                edge_info = np.array([float(poses) for poses in elements[6:]]) 
                info_matrix[0, 0] = edge_info[0]
                info_matrix[0, 1] = info_matrix[1, 0] = edge_info[1]
                info_matrix[0, 2] = info_matrix[2, 0] = edge_info[2]
                info_matrix[1, 1] = edge_info[3]
                info_matrix[1, 2] = info_matrix[2, 1] = edge_info[4]
                info_matrix[2, 2] = edge_info[5]
                cov_matrix = np.linalg.inv(info_matrix)
                values = np.array([float(elements[3]), float(elements[4]), float(elements[5]), cov_matrix])
                edges_dict[key] = values

    return poses_dict, edges_dict

def incremental_solution_2d(poses, edges):
    
    isam = gtsam.ISAM2()
    optimized_poses = []

    for key, value in poses.items():
        factor_graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        x, y, theta = value[0],value[1],value[2]
        idx = int(key)
        pose = gtsam.Pose2(x, y, theta)

        if idx == 0:
            priorNoise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
            factor_graph.add(gtsam.PriorFactorPose2(0, pose, priorNoise));      
            initial_values.insert(idx, pose)

        else: 
            prev_pose = optimized_poses[-1]
            initial_values.insert(idx, prev_pose)

        for key, value in edges.items():
            dx, dy, dtheta, info = value
            ide1, ide2 = int(key[0]), int(key[1])


            if ide2 == idx:
                cov = info
                model = gtsam.noiseModel_Gaussian.Covariance(cov)
                factor_graph.add(gtsam.BetweenFactorPose2(ide1, ide2, gtsam.Pose2(dx, dy, dtheta), model))

        isam.update(factor_graph, initial_values)
        result = isam.calculateEstimate()
        optimized_poses.append(result.atPose2(idx))
    
    return optimized_poses

def plot_trajectory(initial_poses, optimized_poses):
    fig, ax = plt.subplots(figsize=(8, 8))

    plot_values_f = []


    for key, value in initial_poses.items():
        x = value[0]
        y = value[1]
        plot_values_f.append([x,y])

    # plot initial trajectory
    x = [p[0] for p in plot_values_f]
    y = [p[1] for p in plot_values_f]
    ax.plot(x, y, 'r-', label='Initial Trajectory')

    # plot optimized trajectory
    x = [p.x() for p in optimized_poses]
    y = [p.y() for p in optimized_poses]
    ax.plot(x, y, 'g-', label='Optimized Trajectory')

    # set plot attributes
    ax.set_aspect('equal')
    ax.legend(loc='best')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectory Comparison')
    plt.show()

if __name__ == "__main__":
    input_INTEL_g2o = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/input_INTEL_g2o.g2o"
    poses, edges = read_g2o_file_2d(input_INTEL_g2o)

    optimized_poses = incremental_solution_2d(poses, edges)
    # print("optimized",optimized_poses)
    import sys
    # plot the trajectories
    plot_trajectory(poses, optimized_poses)

    
