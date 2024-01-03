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
                values = np.array([float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6]), float(elements[7]),float(elements[8]), float(elements[9]), cov_matrix])
                edges_dict[key] = values

    print(len(poses_dict)) # output 1228
    print(len(edges_dict)) # output 1483
    return poses_dict, edges_dict

if __name__ == "__main__":
    parking_garage = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/parking-garage.g2o"
    x, y = read_g2o_file_3d(parking_garage)
    factor_graph = gtsam.NonlinearFactorGraph()

    priorNoise = gtsam.noiseModel_Gaussian.Covariance(np.eye(6))
    prior_factor = gtsam.Pose3(gtsam.Rot3.Quaternion(1, 0, 0, 0), gtsam.Point3(0, 0, 0))
    factor_graph.add(gtsam.PriorFactorPose3(0, prior_factor, priorNoise))

    initial_values = gtsam.Values()
    for key, value in x.items():
        pose = gtsam.Pose3(gtsam.Rot3.Quaternion(value[6], value[3], value[4],value[5]), gtsam.Point3(value[0], value[1],value[2]))
        initial_values.insert(int(key), pose)


    for key, value in y.items():
        factor = gtsam.BetweenFactorPose3(int(key[0]), int(key[1]), gtsam.Pose3(gtsam.Rot3.Quaternion(value[6], value[3], value[4],value[5]), gtsam.Point3(value[0], value[1],value[2])), gtsam.noiseModel_Gaussian.Covariance(value[7]))
        factor_graph.add(factor)

    optimizer = gtsam.GaussNewtonOptimizer(factor_graph, initial_values)
    result = optimizer.optimizeSafely()


    unoptimized_poses = []
    for key, value in x.items():
        x = value[0]
        y = value[1]
        z = value[2]
        val = np.array([x,y,z])
        unoptimized_poses.append(val)
    

    plot_values_f = []

    for i in range(result.size()):
        plot_values = np.array([result.atPose3(i).x(), result.atPose3(i).y(), result.atPose3(i).z()])
        plot_values_f.append(plot_values)


    plot_values_f = np.array(plot_values_f)
    unoptimized_poses = np.array(unoptimized_poses)
    ax = plt.figure().add_subplot(projection = '3d')
    ax.plot(unoptimized_poses[:,0], unoptimized_poses[:,1], unoptimized_poses[:,2], label='Unoptimized trajectory')
    ax.plot(plot_values_f[:,0],plot_values_f[:,1], plot_values_f[:,2], label='Optimized trajectory')
    ax.legend()
    plt.show()
                
    