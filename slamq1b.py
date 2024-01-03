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
                values = np.array([float(elements[3]), float(elements[4]), float(elements[5]), cov_matrix])
                edges_dict[key] = values

    print(len(poses_dict)) # output 1228
    print(len(edges_dict)) # output 1483
    return poses_dict, edges_dict



if __name__ == "__main__":
    input_INTEL_g2o = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/input_INTEL_g2o.g2o"
    x, y = read_g2o_file_2d(input_INTEL_g2o)


    factor_graph = gtsam.NonlinearFactorGraph()
    print(type(factor_graph))

    priorNoise = gtsam.noiseModel_Gaussian.Covariance(np.eye(3))
    factor_graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), priorNoise)); 

    initial_values = gtsam.Values()
    for key, value in x.items():
        pose = gtsam.Pose2(value[0],value[1],value[2])
        initial_values.insert(int(key), pose)

    for key, value in y.items():
        # if ()
        # sigmas = [value[3][i, i] for i in range(3)]
        noise_model = gtsam.noiseModel_Gaussian.Covariance(value[3])
        pose = gtsam.Pose2(value[0], value[1], value[2])
        factor_graph.add(gtsam.BetweenFactorPose2(int(key[0]), int(key[1]), pose, noise_model))
        
    optimizer = gtsam.GaussNewtonOptimizer(factor_graph, initial_values)
    result = optimizer.optimize()
    print("\nFinal result:\n")
    print(result)

    unoptimized_poses = []
    for key, value in x.items():
        x = value[0]
        y = value[1]
        val = np.array([x,y])
        unoptimized_poses.append(val)
    
    plot_values_f = []

    for i in range(result.size()):
        plot_values = np.array([result.atPose2(i).x(), result.atPose2(i).y()])
        plot_values_f.append(plot_values)

    plot_values_f = np.array(plot_values_f)
    unoptimized_poses = np.array(unoptimized_poses)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(unoptimized_poses[:,0], unoptimized_poses[:,1], label='Unoptimized trajectory')
    ax.plot(plot_values_f[:,0],plot_values_f[:,1], label='Optimized trajectory')
    ax.legend()
    plt.show()
    
