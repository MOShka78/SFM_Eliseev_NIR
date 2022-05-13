import open3d as o3d
import cv2
import os
import yaml
import open3d as o3d
def civ():
    with open('struc.yaml') as fh:
        read_data = yaml.load(fh, Loader=yaml.FullLoader)
        print(read_data)
    Point = read_data['Points']
    print(Point[1][0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(read_data['Points'])
    o3d.visualization.draw_geometries([pcd])


def extractionVideo():

    cam = cv2.VideoCapture("/home/vadim/NIR/video.mp4")
    if not os.path.exists('photo00'):
        os.makedirs('photo00')

    currentframe = 0
    flag = 0

    while (True):
        ret, img = cam.read()
        if ret:
            if flag == 42:
                name = './photo00/exper' + str(currentframe) + '.jpg'
                cv2.imwrite(name, img)
                currentframe += 1
                flag = 0
            else:
                flag = flag + 1
        else:
            break

    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    civ()