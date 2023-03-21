
import numpy as np
import os
import glob
import pandas as pd




def sync_file_gen(path_,name='test_sync'):
    '''
    This function generate 
    '''
    # path_= r'datasets/rgbd_dataset_freiburg1_xyz_clear_test'

    files=glob.glob(path_+'\*')
    for i in range(0,len(files)):
        files[i]=files[i][47:]

    # print(files)


    for i in files[2],files[6]:
        globals()['%s_data'%i[:-4]] = pd.read_csv(path_+"/"+i, delim_whitespace=True, error_bad_lines=False,header=None,skiprows=3)
        globals()['%s_data'%i[:-4]].columns = ["timestamp", "path"]

    groundtruth_data=pd.read_csv(path_+"/"+files[3], delim_whitespace=True, error_bad_lines=False,header=None,skiprows=3)
    groundtruth_data.columns = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]


    res=np.zeros(len(depth_data.timestamp))

    for j in range(0,len(depth_data.timestamp) - 1):
        error=1e13
        for i in range(0,len(groundtruth_data.timestamp) - 1):
            # print("error= ",np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j]))
            if np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j])<error:
                error=np.abs(groundtruth_data.timestamp[i] - depth_data.timestamp[j])
                # print('j=',j,' i=',i," error=",error,'\n')
                res[j]=i        


    with open(path_+"/"+name+'.txt', 'w') as f:
        f.write("timestamp depth_path rgb_path \n")
        for i in range(0,len(res)-1) :
            f.write(str(groundtruth_data.timestamp[int(res[i])])+' '+str(depth_data.path[i])+' '+str(rgb_data.path[i]))
            f.write('\n')
    
    #ground_true gen
    

    with open(path_+"/"+"groundtruth_sync"+'.txt', 'w') as f:
        f.write("timestamp tx ty tz qx qy qz qw \n")
        for i in range(0,len(res)-1) :

            tx=(groundtruth_data.tx[int(res[i])])
            ty=(groundtruth_data.ty[int(res[i])])
            tz=(groundtruth_data.tz[int(res[i])])
            qx=(groundtruth_data.qx[int(res[i])])
            qy=(groundtruth_data.qy[int(res[i])])
            qz=(groundtruth_data.qz[int(res[i])])
            qw=(groundtruth_data.qw[int(res[i])])
            pose=np.array([tx,ty,tz,qx,qy,qz,qw])

            f.write(pose[0]+' '+pose[1]+' '+pose[2]+' '+pose[3]+' '+pose[4]+' '+pose[5]+' '+pose[6])
            f.write('\n')

def get_pose(i):
    # with open(path_+"/"+"groundtruth_sync"+'.txt', 'w') as f:
    tx=(groundtruth_data.tx[int(res[i])])
    ty=(groundtruth_data.ty[int(res[i])])
    tz=(groundtruth_data.tz[int(res[i])])
    qx=(groundtruth_data.qx[int(res[i])])
    qy=(groundtruth_data.qy[int(res[i])])
    qz=(groundtruth_data.qz[int(res[i])])
    qw=(groundtruth_data.qw[int(res[i])])
    pose=np.array([tx,ty,tz,qx,qy,qz,qw])
    return pose

