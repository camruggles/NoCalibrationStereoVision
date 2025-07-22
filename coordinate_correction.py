import numpy as np


class AffineTransformer:
    @staticmethod
    def get_affine_transform_from_points(src,dst):
        '''
        given minimum 4 points from 2 different coordinate systems
        estimate the affine transformation that would map src to dst using
        least squares to estimate the coordinates of the affine transform matrix

        ARGS:
        src: np.array of size nx3 in the first coordinate system
        dst: np.array of size nx3 in the second coordinate system

        RETURNS:
        T: np.array of size 4x4, the affine transform that will map homogenous coordinates from
            the first to the second coordinate system

        '''

        n = src.shape[0]
        M = np.zeros((3*n, 12))
        b = np.zeros((3*n, 1))

        # import pdb
        # pdb.set_trace()

        for i, ((x,y,z),(xp,yp,zp)) in enumerate(zip(src, dst)):
            M[3*i]   = [x,y,z,1, 0,0,0,0, 0,0,0,0]
            M[3*i+1] = [0,0,0,0, x,y,z,1, 0,0,0,0]
            M[3*i+2] = [0,0,0,0, 0,0,0,0, x,y,z,1]
            b[3*i]   = xp
            b[3*i+1] = yp
            b[3*i+2] = zp

        theta, *_ = np.linalg.lstsq(M, b, rcond=None)
        a11,a12,a13,tx,a21,a22,a23,ty,a31,a32,a33,tz = theta.flatten()

        A = np.array([[a11,a12,a13],
                    [a21,a22,a23],
                    [a31,a32,a33]])
        t = np.array([tx,ty,tz])

        # Homogeneous 4x4 matrix
        T = np.eye(4)
        T[:3,:3] = A
        T[:3, 3] = t
        print('estimated' , T.shape)

        return T

    @staticmethod
    def apply_transform(T, points):
        '''
        need this function to turn an nx3 point into a 4xn vector representating homogenous coordinates

        args:
        points : nx3 numpy array with source points

        '''

        points = points.T
        ones_row = np.ones((1,points.shape[1]))
        points_homogenous = np.vstack((points, ones_row)) # shape 4xn
        ret = np.matmul(T, points_homogenous)
        return ret



if __name__ == "__main__":
    
    # source (before) and destination (after) points as Nx3 arrays
    # src = np.array([[x1,y1,z1],
    #                 [x2,y2,z2],
    #                 [x3,y3,z3],
    #                 [x4,y4,z4]], dtype=float)
    src = np.random.rand(4,3)
    # dst = np.array([[x1p,y1p,z1p],
    #                 [x2p,y2p,z2p],
    #                 [x3p,y3p,z3p],
    #                 [x4p,y4p,z4p]], dtype=float)

    dst = np.random.rand(4,3)
    T = AffineTransformer.get_affine_transform_from_points(src,dst)
    dst_hat = AffineTransformer.apply_transform(T, src)
    # import pdb
    # pdb.set_trace()
    # print(dst)
    # print(np.matmul(T,src))
    print(dst_hat)
    print(dst.T)
    print(dst_hat[:3, :] - dst.T)

# how can you use Cursor to prompt this into existence
# change file and function organization later to increase code locality