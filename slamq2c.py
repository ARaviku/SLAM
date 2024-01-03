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

def incremental_solution_3d(poses, edges):
    isam = gtsam.ISAM2()
    optimized_poses = []

    for key, value in poses.items():
        factor_graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()

        x, y, z, qw, qx, qy, qz= value[0],value[1],value[2],value[6], value[3],value[4], value[5] 
        idx = int(key)

        if idx == 0:
            priorNoise = gtsam.noiseModel_Gaussian.Covariance(np.eye(6))
            pose = gtsam.Pose3(gtsam.Rot3.Quaternion(qw, qx, qy, qz), gtsam.Point3(x, y, z))
            factor_graph.add(gtsam.PriorFactorPose3(0, pose, priorNoise));      
            initial_values.insert(idx, pose)

        else: 
            prev_pose = optimized_poses[idx -1]
            initial_values.insert(idx, prev_pose)

        for key, value in edges.items():
            dx, dy, dz, dqw, dqx, dqy, dqz, info = value[0],value[1],value[2],value[6], value[3],value[4], value[5], value[7]
            ide1, ide2 = int(key[0]), int(key[1])


            if ide2 == idx:
                cov = info
                model = gtsam.noiseModel_Gaussian.Covariance(cov)
                pose = gtsam.Pose3(gtsam.Rot3.Quaternion(dqw, dqx, dqy, dqz), gtsam.Point3(dx, dy, dz))
                factor_graph.add(gtsam.BetweenFactorPose3(ide1, ide2, pose, model))

        isam.update(factor_graph, initial_values)
        result = isam.calculateEstimate()
        optimized_poses.append(result.atPose3(idx))
    
    return optimized_poses

def plot_trajectory(initial_poses, optimized_poses):
    
    unoptimized_poses = []
    for key, value in initial_poses.items():
        x = value[0]
        y = value[1]
        z = value[2]
        val = np.array([x,y,z])
        unoptimized_poses.append(val)
    

    plot_values_f = []

    for pose in optimized_poses:
        plot_values = np.array([pose.x(), pose.y(), pose.z()])
        plot_values_f.append(plot_values)


    plot_values_f = np.array(plot_values_f)
    unoptimized_poses = np.array(unoptimized_poses)
    ax = plt.figure().add_subplot(projection = '3d')
    ax.plot(unoptimized_poses[:,0], unoptimized_poses[:,1], unoptimized_poses[:,2], label='Unoptimized trajectory')
    ax.plot(plot_values_f[:,0],plot_values_f[:,1], plot_values_f[:,2], label='Optimized trajectory')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parking_garage = "/home/annu/Desktop/Project_codes/anurekha_hw-slam/parking-garage.g2o"
    poses, edges = read_g2o_file_3d(parking_garage)
    import sys
    optimized_poses = incremental_solution_3d(poses, edges)
    plot_trajectory(poses, optimized_poses)

    
