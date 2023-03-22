import numpy as np
import os
import glob
import pandas as pd


class Sync:
    """
    # example of using
    xyz_ds=Sync()
    xyz_ds.init(path_xyz=r"datasets/rgbd_dataset_freiburg1_xyz_clear_test")
    xyz_ds.sort()
    xyz_ds.write_txt_sync()
    xyz_ds.write_txt_ground_truth()
    xyz_ds.get_ground_truth_sync(50)
    """

    def init(self, path_xyz):
        """
        reading data from .txt files
        args:
        path_xyz- path of directory, where .txt files is located
        """
        ###
        self.path_xyz = path_xyz

        files = glob.glob(path_xyz + "\*")
        for i in range(0, len(files)):
            files[i] = files[i][47:]

        print(files)
        ###
        self.depth_data = pd.read_csv(
            path_xyz + "/" + "depth.txt", delim_whitespace=True, header=None, skiprows=3
        )
        self.depth_data.columns = ["timestamp", "path"]
        ###
        self.rgb_data = pd.read_csv(
            path_xyz + "/" + "rgb.txt", delim_whitespace=True, header=None, skiprows=3
        )
        self.rgb_data.columns = ["timestamp", "path"]
        ###
        self.groundtruth_data = pd.read_csv(
            path_xyz + "/" + "groundtruth.txt",
            delim_whitespace=True,
            header=None,
            skiprows=3,
        )
        self.groundtruth_data.columns = [
            "timestamp",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
        ]
        ###

    def sort(self):
        """
        synchronize timestemps between groundtruth and depth
        """
        self.res = np.zeros(len(self.depth_data.timestamp))

        for j in range(0, len(self.depth_data.timestamp) - 1):
            error = 1e13
            for i in range(0, len(self.groundtruth_data.timestamp) - 1):
                # print("error= ",np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j]))
                if (
                    np.abs(
                        self.groundtruth_data.timestamp[i]
                        - self.depth_data.timestamp[j]
                    )
                    < error
                ):
                    error = np.abs(
                        self.groundtruth_data.timestamp[i]
                        - self.depth_data.timestamp[j]
                    )
                    # print('j=',j,' i=',i," error=",error,'\n')
                    self.res[j] = i

    def sort_fast(self):
        """
        synchronize timestemps between groundtruth and depth
        """
        self.res = np.zeros(len(self.depth_data.timestamp))
        
        for j in range(0, len(self.depth_data.timestamp) - 1):
            error = 1e13
            if j==0: 
                for i in range(0, len(self.groundtruth_data.timestamp) - 1):
                    # print("error= ",np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j]))
                    if (np.abs(self.groundtruth_data.timestamp[i] - self.depth_data.timestamp[j]) < error):
                        error = np.abs(
                            self.groundtruth_data.timestamp[i]
                            - self.depth_data.timestamp[j]
                        )
                        start_pos=i
            else:
                counter=0
                for i in range(start_pos, len(self.groundtruth_data.timestamp) - 1):
                    # print("error= ",np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j]))
                    if (np.abs(self.groundtruth_data.timestamp[i] - self.depth_data.timestamp[j]) < error):
                        counter+=1
                      
                        error = np.abs(
                            self.groundtruth_data.timestamp[i]
                            - self.depth_data.timestamp[j]
                        )
                        if counter>8:
                            break
                        self.res[j] = i
                    # print('j=',j,' i=',i," error=",error,'\n')
                    

    def write_txt_sync(self):
        """
        create .txt files, which has associated data such as timestemp,path to depth image and path to rgb image
        """
        with open(self.path_xyz + "/" + "test_sync_fast.txt", "w") as f:
            f.write("timestamp depth_path rgb_path \n")
            for i in range(0, len(self.res) - 1):
                f.write(
                    str(self.groundtruth_data.timestamp[int(self.res[i])])
                    + " "
                    + str(self.depth_data.path[i])
                    + " "
                    + str(self.rgb_data.path[i])
                )
                f.write("\n")

    def write_txt_ground_truth(self):
        """
        create .txt file with synhronize timestemp
        """
        with open(self.path_xyz + "/" + "ground_truth_sync.txt", "w") as f:
            f.write("timestamp tx ty tz qx qy qz qw \n")
            for i in range(0, len(self.res) - 1):
                timestamp = self.groundtruth_data.timestamp[int(self.res[i])]
                tx = self.groundtruth_data.tx[int(self.res[i])]
                ty = self.groundtruth_data.ty[int(self.res[i])]
                tz = self.groundtruth_data.tz[int(self.res[i])]
                qx = self.groundtruth_data.qx[int(self.res[i])]
                qy = self.groundtruth_data.qy[int(self.res[i])]
                qz = self.groundtruth_data.qz[int(self.res[i])]
                qw = self.groundtruth_data.qw[int(self.res[i])]
                pose = np.array([timestamp, tx, ty, tz, qx, qy, qz, qw])

                f.write(
                    f"{pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]} {pose[7]}\n"
                )

    def get_ground_truth_sync(self, i: int):
        """
        get pose on current timestemp

        Args:
            i (int): the number of timestemp
        """
        tx = self.groundtruth_data.tx[int(self.res[i])]
        ty = self.groundtruth_data.ty[int(self.res[i])]
        tz = self.groundtruth_data.tz[int(self.res[i])]
        qx = self.groundtruth_data.qx[int(self.res[i])]
        qy = self.groundtruth_data.qy[int(self.res[i])]
        qz = self.groundtruth_data.qz[int(self.res[i])]
        qw = self.groundtruth_data.qw[int(self.res[i])]
        pose = np.array([tx, ty, tz, qx, qy, qz, qw])
        return pose
