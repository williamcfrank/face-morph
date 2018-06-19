# pylint: disable=W,C

import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.spatial import tsearch
from scipy.interpolate import interp2d
# from scipy.sparse.linalg import lsqr
import numpy as np
import numpy.linalg as linalg
import sys

def getPoints(im0, im1):
    # print("Select inputs for face0")
    # plt.imshow(face0)
    # tri0_vertices_raw = plt.ginput(-1, timeout=0, show_clicks=True)
    # plt.close()
    # print("Select inputs for face1")
    # plt.imshow(face1)
    # tri1_vertices_raw = plt.ginput(-1, timeout=0, show_clicks=True)
    # plt.close()
    tri0_vertices_raw = [(777.33549783549802, 548.07142857142844), (773.87229437229462, 703.91558441558436), (735.77705627705654, 797.42207792207785), (586.85930735930742, 939.41341991341983), (521.0584415584417, 949.80303030303025), (431.01515151515173, 935.95021645021643), (316.72943722943728, 818.2012987012987), (254.39177489177496, 717.76839826839819), (240.53896103896113, 579.24025974025972), (237.07575757575762, 423.3961038961038), (299.41341991341994, 260.62554112554108), (493.35281385281405, 184.4350649350647), (645.73376623376635, 212.14069264069258), (773.87229437229462, 399.15367965367955), (732.31385281385292, 426.85930735930719), (649.19696969696997, 423.3961038961038), (566.0800865800868, 440.71212121212113), (420.6255411255413, 461.49134199134187), (351.36147186147195, 437.24891774891762), (282.09740259740272, 451.10173160173156), (708.07142857142867, 516.90259740259728), (638.80735930735955, 537.68181818181813), (583.39610389610402, 527.29220779220771), (635.34415584415592, 492.66017316017303), (420.6255411255413, 541.14502164502153), (379.0670995670996, 548.07142857142844), (289.02380952380963, 537.68181818181813), (372.14069264069281, 509.97619047619037), (635.34415584415592, 520.36580086580079), (372.14069264069281, 534.21861471861462), (590.32251082251105, 679.67316017316011), (507.20562770562788, 703.91558441558436), (424.0887445887447, 686.59956709956703), (503.74242424242448, 651.96753246753235), (500.27922077922085, 568.85064935064929), (638.80735930735955, 773.1796536796536), (566.0800865800868, 825.12770562770561), (472.5735930735932, 825.12770562770561), (392.91991341991343, 776.64285714285711), (493.35281385281405, 755.86363636363626), (545.30086580086595, 755.86363636363626), (902.01082251082266, 849.37012987012986), (150.49567099567105, 894.39177489177484)]
    tri1_vertices_raw = [(708.07142857142867, 593.09307359307354), (701.14502164502187, 662.35714285714278), (690.75541125541145, 735.08441558441552), (548.76406926406935, 890.92857142857144), (469.1103896103898, 904.78138528138527), (379.0670995670996, 890.92857142857144), (257.85497835497847, 766.25324675324669), (237.07575757575762, 707.37878787878788), (233.61255411255422, 610.40909090909088), (247.46536796536805, 433.78571428571422), (295.95021645021654, 305.64718614718595), (489.88961038961043, 208.67748917748895), (645.73376623376635, 298.72077922077915), (697.68181818181824, 444.17532467532453), (663.0497835497838, 409.54329004328997), (597.24891774891785, 385.30086580086572), (517.5952380952383, 385.30086580086572), (403.30952380952385, 399.15367965367955), (347.89826839826844, 395.69047619047615), (282.09740259740272, 409.54329004328997), (645.73376623376635, 482.27056277056272), (576.46969696969722, 499.58658008658006), (531.44805194805213, 485.73376623376612), (590.32251082251105, 451.10173160173156), (406.77272727272748, 496.12337662337654), (358.28787878787887, 506.51298701298697), (275.17099567099569, 489.19696969696963), (358.28787878787887, 454.56493506493496), (583.39610389610402, 475.34415584415581), (361.75108225108238, 482.27056277056272), (548.76406926406935, 641.57792207792204), (472.5735930735932, 641.57792207792204), (389.45670995671003, 627.7251082251081), (469.1103896103898, 589.62987012987003), (469.1103896103898, 513.43939393939388), (586.85930735930742, 731.62121212121201), (514.13203463203467, 762.79004329004329), (431.01515151515173, 762.79004329004329), (358.28787878787887, 735.08441558441552), (437.94155844155853, 703.91558441558436), (489.88961038961043, 703.91558441558436), (884.69480519480544, 970.5822510822511), (108.93722943722946, 960.19264069264068)]

    return tri0_vertices_raw, tri1_vertices_raw

def showImg(im):
    plt.imshow(im)
    plt.show()

def plotTriangulation(im, triangulations):
    plt.triplot(im[:,0], im[:,1], triangulations.copy())
    plt.plot(im[:,0], im[:,1], 'o')
    plt.show()

def ginput_to_array(g_in):
    lst = []
    for point in g_in:
        lst.append([int(point[0]), int(point[1])])
    return np.array(lst)

def ginput_to_array_other(g_in):
    lst = []
    for point in g_in:
        lst.append([int(point[0]), int(point[1])])
    return lst

def meanPointSet(v_1, v_2):
    v = []
    for y in range(len(v_1)):
        v.append([(v_1[y][0] + v_2[y][0])/2, (v_1[y][1] + v_2[y][1])/2])
    return ginput_to_array(v)

def appendCorners(im, pts):
    pts.append((0, im.shape[0] - 1)) # bottom left corner
    pts.append((0,0)) # top left
    pts.append((im.shape[1] - 1, 0)) # top right
    pts.append((im.shape[1] - 1, im.shape[0] - 1)) # bottom right
    return pts

def appendCorners_other(im, pts):
    pts.append((0,0)) # top left
    pts.append((im.shape[1] - 1, 0)) # top right
    pts.append((0, im.shape[0] - 1)) # bottom left corner
    pts.append((im.shape[1] - 1, im.shape[0] - 1)) # bottom right
    return pts

def computeAffine(tri0_vertices, tri1_vertices, tri_vertices_indeces):
    # Create X using tri0_vertices
    X = np.zeros((6,6))
    for i in range(3):
        tri_vertex = tri_vertices_indeces[i]
        # print(tri_vertex)
        # print(tri0_vertices)
        x_i = tri0_vertices[tri_vertex][0]
        y_i = tri0_vertices[tri_vertex][1]
        X[2*i, :] = [x_i,y_i,1,0,0,0]
        X[2*i+1, :] = [0,0,0,x_i,y_i,1]

    # Create x`
    x_prime = np.zeros((6,1))
    for i in range(3):
        tri_vertex = tri_vertices_indeces[i]
        x_i_prime = tri1_vertices[tri_vertex][0]
        y_i_prime = tri1_vertices[tri_vertex][1]
        x_prime[2*i, :] = x_i_prime
        x_prime[2*i+1, :] = y_i_prime
    # Solve for a
    a = linalg.lstsq(X, x_prime)[0]
    # Create affine transformation matrix
    A = np.zeros((3,3))
    A[0,:] = [a[0], a[1], a[2]]
    A[1,:] = [a[3], a[4], a[5]]
    A[2,:] = [0,0,1]
    return A

def computeAffines(triangulations, src_vertices, dst_vertices):
    affine_matrices = []
    i = 0
    for tri in triangulations:
        A = computeAffine(src_vertices, dst_vertices, tri)
        affine_matrices.append(A)
        i += 1
    return affine_matrices

def warp(h, w, t, affine_matrices_inv, src):
    # (Inverse) Warp each pixel
    mid = np.zeros((h,w,3))
    for i in range(h):
        for j in range(w): # added here to see if i still get index errors
            tri_vertices_indeces_index = tsearch(t,[j,i])
            iw = inverse_warp(affine_matrices_inv[tri_vertices_indeces_index], j, i)
            x = int(iw[0])
            y = int(iw[1])
            if x > w - 1:
                print("Inverse Warp returned an x greater than width.")
                print(x)
                x = w - 1
            if y > h -1:
                print("Inverse Warp returned a y greater than height.")
                print(y)
                y = h - 1
            if x < 0:
                print("Inverse Warp returned a x less than 0.")
                print(x)
                x = 0
            if y < 0:
                print("Inverse Warp returend a  y less than 0.")
                print(y)
                y = 0

            mid[i, j, :] = src[y, x, :]
    return mid

def inverse_warp(A_inv, x, y):
    x_vector = np.zeros((3,1))
    x_vector[0] = x
    x_vector[1] = y
    x_vector[2] = 1
    return np.dot(A_inv, x_vector)

def createMidway(face0, face1, h, w, tri0_vertices, tri1_vertices):
    # Compute the Midway-Face
    # 1) Compute the average shape (a.k.a the average of each keypoint location in the two faces)
    # 2) Warp both faces into that shape
    # 3) Average the colors together.

    # Create Dalaunay triangulation at the midway set
    mean_vertices = meanPointSet(tri0_vertices, tri1_vertices)
    t = Delaunay(mean_vertices)
    trianguations = t.simplices

    # Compute Affine Transformation matrices for both transformations (src-->mid; dst-->mid)
    affine_matrices_0 = computeAffines(trianguations, tri0_vertices, mean_vertices)
    affine_matrices_inv_0 = [linalg.inv(A) for A in affine_matrices_0]
    affine_matrices_1 = computeAffines(trianguations, tri1_vertices, mean_vertices)
    affine_matrices_inv_1 = [linalg.inv(A) for A in affine_matrices_1]

    # Midway-Faces
    mid0 = warp(h, w, t, affine_matrices_inv_0, face0)
    mid1 = warp(h, w, t, affine_matrices_inv_1, face1)

    # Average the colors together
    frac = .5
    mid = mid0.astype(np.float32) * frac + mid1.astype(np.float32) * (1-frac)
    return mid.astype(np.uint8)

# def morph_frame(face0, face1, h, w, tri0_vertices, tri1_vertices, t, warp_frac, dissolve_frac):

def morphPointSet(v_1, v_2, warp_frac):
    """
    Creates point set for the ith warp
    """
    v = []
    for y in range(len(v_1)):
        v.append([(v_1[y][0] * (1 - warp_frac) + v_2[y][0] * warp_frac) , (v_1[y][1] * (1 - warp_frac) + v_2[y][1] * warp_frac)])
    return ginput_to_array(v)

def morph_frame(mid0, mid1, dissolve_frac):
    """
    Applies warp fraction [0,1] and dissolve fraction [0,1] to a single mid frame.
    """
    frame = mid0.astype(np.float32) * (1 - dissolve_frac) + mid1.astype(np.float32) * dissolve_frac
    return frame.astype(np.uint8)


def morph(face0, face1, h, w, tri0_vertices, tri1_vertices):
    """
    Main controller function for creating a full sequence morph.
    """
    frames = []

    for i in range(46):
        # i = 5
        print("Working on the " + str(i) + "th frame!")
        print("Creating triangulation.")
        # Create Dalaunay triangulation for the ith morph set
        morph_verticies = morphPointSet(tri0_vertices, tri1_vertices, i/45)
        t = Delaunay(morph_verticies)
        trianguations = t.simplices

        # Compute Affine Transformation matrices for both transformations (src-->mid; dst-->mid)
        print("Computing Affine Transformation.")
        affine_matrices_0 = computeAffines(trianguations, tri0_vertices, morph_verticies)
        affine_matrices_inv_0 = [linalg.inv(A) for A in affine_matrices_0]
        affine_matrices_1 = computeAffines(trianguations, tri1_vertices, morph_verticies)
        affine_matrices_inv_1 = [linalg.inv(A) for A in affine_matrices_1]

        # Morphed images
        print("Inverse warping.")
        morph0 = warp(h, w, t, affine_matrices_inv_0, face0)
        morph1 = warp(h, w, t, affine_matrices_inv_1, face1)
        # frame = morph_frame(face0, face1, h, w, tri0_vertices, tri1_vertices, t, i/45, i/45)

        print("Creating morph frame.")
        frame = morph_frame(morph0, morph1, i/45)
        frames.append(frame)
        # break

    return frames

def save_gif(frames):
    """
    Saves a list of images for the annimated gif.
    """
    print("Saving gif images!")
    for i in range(len(frames)):
        im_out_path = "gif/gif_emilie_will_" + str(i) + ".png"
        plt.imsave(im_out_path, frames[i])

def get_will_points():
    h = 480
    w = 640
    my_points = [(205.21428571428569, 314.30519480519479), (209.1103896103896, 345.47402597402595), (211.70779220779221, 370.14935064935059), (231.1883116883117, 402.61688311688306), (253.26623376623374, 419.5), (277.94155844155841, 435.08441558441552), (311.70779220779218, 444.17532467532465), (341.57792207792204, 438.98051948051943), (359.75974025974028, 425.99350649350646), (383.13636363636363, 407.81168831168827), (409.11038961038957, 370.14935064935059), (422.09740259740261, 340.27922077922074), (425.99350649350652, 315.60389610389609), (400.01948051948057, 237.68181818181813), (385.73376623376623, 229.88961038961031), (368.85064935064941, 224.6948051948051), (354.56493506493507, 224.6948051948051), (337.68181818181824, 236.38311688311683), (350.66883116883116, 245.47402597402589), (364.9545454545455, 246.7727272727272), (381.83766233766232, 245.47402597402589), (222.09740259740258, 240.27922077922074), (231.1883116883117, 229.88961038961031), (248.07142857142858, 225.9935064935064), (263.65584415584419, 229.88961038961031), (279.24025974025972, 240.27922077922074), (264.9545454545455, 248.0714285714285), (249.37012987012989, 251.96753246753241), (235.08441558441555, 246.7727272727272), (337.68181818181824, 201.31818181818176), (350.66883116883116, 198.72077922077915), (372.74675324675331, 190.92857142857133), (392.22727272727275, 196.12337662337654), (407.81168831168827, 203.91558441558436), (284.43506493506493, 210.40909090909082), (267.5519480519481, 202.61688311688306), (249.37012987012989, 200.01948051948045), (231.1883116883117, 200.01948051948045), (215.60389610389612, 205.21428571428567), (261.05844155844159, 361.05844155844147), (283.13636363636363, 342.87662337662334), (310.40909090909088, 337.68181818181813), (336.38311688311694, 344.17532467532465), (358.46103896103898, 364.95454545454538), (328.59090909090912, 379.24025974025972), (305.21428571428567, 381.83766233766232), (280.53896103896102, 376.64285714285711), (275.34415584415581, 290.92857142857139), (288.33116883116884, 277.94155844155836), (297.42207792207796, 255.86363636363632), (275.34415584415581, 311.70779220779218), (293.52597402597405, 313.00649350649348), (310.40909090909088, 313.00649350649348), (328.59090909090912, 313.00649350649348), (344.17532467532465, 309.11038961038957), (342.87662337662334, 290.92857142857139), (335.08441558441564, 275.34415584415581), (325.99350649350652, 259.75974025974017)]

    my_points = ginput_to_array_other(my_points)
    my_points.append([0,0])
    my_points.append([w-1, 0])
    my_points.append([0, h-1])
    my_points.append([w-1, h-2])

    return np.array(my_points)

def file_parser(file_name):
    """
    Reads predefined corrispondence and returns array of [x,y]
    """
    h = 480
    w = 640
    out = []
    with open(file_name, 'r') as f:
        line_num = 1
        for line in f:
            if line_num < 17:
                # Read to where data starts
                line_num += 1
                continue
            elif line_num > 74:
                break
            # print(list(map(int, line.strip().split(" "))))
            vals = line.split()
            # print(list("".join(line)))
            # print(line.split())
            assert(float(vals[2]) < 640)
            assert(float(vals[3]) < 480)
            point = [float(vals[2]) * w, float(vals[3]) * h]
            # print(point)
            out.append(point)
            line_num += 1

    out.append([0,0])
    out.append([w-1, 0])
    out.append([0, h-1])
    out.append([w-1, h-2])
    return out

def population_text_parser():
    """
    Main controller for mean faces.
    """
    h = 480
    w = 640
    filenames = []
    # get corrispondence ppoints for all faces
    all_face_verticies = []
    for i in range(1, 41):
        if i == 2 or i == 3 or i == 4:
            # not in data set
            continue
        # get file name
        file_name = "data/"
        if i < 10:
            file_name += "0" + str(i)
        else:
            file_name += str(i)
        if i == 8 or i == 22 or i == 30 or i == 35 or i == 12 or i ==14 or i == 15:
            # female faces
            file_name += "-1f.asf"
        else:
            file_name += "-1m.asf"
        filenames.append(file_name[:-3] + "bmp")
        # Parse corrispondence points
        face_vertices = file_parser(file_name)
        all_face_verticies.append(np.array(face_vertices))

    # adding my face to the set
    # all_face_verticies.append(get_will_points())
    mean_vertices = np.array(sum(all_face_verticies)) / len(all_face_verticies)

    # face = plt.imread(filenames[0])
    # mean_vertices = ginput_to_array(appendCorners_other(face, mean_vertices))
    # Morph each of the faces in the dataset into the average shape.
    morphs = []
    for i in range(len(all_face_verticies)):
        # Read in src img
        print("Morphing face " + str(i) + " into the average shape.")
        if i == len(all_face_verticies)-1:
            face = plt.imread("will_population.jpeg")/255
        else:
            face = plt.imread(filenames[i])/255
        # print(face.shape)
        # print(mean_vertices)
        im_src_vertices = all_face_verticies[i]

        # Create Dalaunay triangulation for the ith morph set
        print("Computing Delaunay triangulation.")
        # morph_verticies = morphPointSet(tri0_vertices, tri1_vertices, i/45)
        t = Delaunay(mean_vertices)
        trianguations = t.simplices

        # Compute Affine Transformation matrices for both transformations (src-->mid; dst-->mid)
        print("Computing Affine Transformation.")
        affine_matrices_0 = computeAffines(trianguations, im_src_vertices, mean_vertices)
        affine_matrices_inv_0 = [linalg.inv(A) for A in affine_matrices_0]

        morph = warp(h, w, t, affine_matrices_inv_0, face)
        # if i == len(all_face_verticies) -1:
        #     plt.imsave("will_in_danish_pop.png", morph)
        morphs.append(morph)

    out = sum(morphs) * (1/len(morphs))
    return out

def getAvgDaneShape():
    h = 480
    w = 640
    # get corrispondence ppoints for all faces
    all_face_verticies = []
    for i in range(1, 41):
        if i == 2 or i == 3 or i == 4:
            # not in data set
            continue
        # get file name
        file_name = "data/"
        if i < 10:
            file_name += "0" + str(i)
        else:
            file_name += str(i)
        if i == 8 or i == 22 or i == 30 or i == 35 or i == 12 or i ==14 or i == 15:
            # female faces
            file_name += "-1f.asf"
        else:
            file_name += "-1m.asf"
        # Parse corrispondence points
        face_vertices = file_parser(file_name)
        all_face_verticies.append(np.array(face_vertices))

    # adding my face to the set
    # all_face_verticies.append(get_will_points())
    mean_vertices = np.array(sum(all_face_verticies)) / len(all_face_verticies)
    return mean_vertices

def getAvgFemaleShape():
    h = 480
    w = 640
    # get corrispondence ppoints for all faces
    all_face_verticies = []
    for i in range(1, 41):
        if i == 2 or i == 3 or i == 4:
            # not in data set
            continue
        # get file name
        file_name = "data/"
        if i < 10:
            file_name += "0" + str(i)
        else:
            file_name += str(i)
        if i == 8 or i == 22 or i == 30 or i == 35 or i == 12 or i ==14 or i == 15:
            # female faces
            file_name += "-1f.asf"
        else:
            continue
        # Parse corrispondence points
        face_vertices = file_parser(file_name)
        all_face_verticies.append(np.array(face_vertices))

    # adding my face to the set
    # all_face_verticies.append(get_will_points())
    mean_vertices = np.array(sum(all_face_verticies)) / len(all_face_verticies)
    return mean_vertices

def makeCaricature(scalar):
    h = 480
    w = 640
    my_face = plt.imread("will_population.jpeg")/255
    # avg_dane = plt.imread("mean_danish_face.png")/255
    mean_vertices = getAvgDaneShape() # np array
    im_src_vertices = get_will_points() # np array

    caricature_vertices = scalar * (im_src_vertices - mean_vertices) + mean_vertices

    print("Computing Delaunay triangulation.")
    t = Delaunay(mean_vertices)
    trianguations = t.simplices

    # Compute Affine Transformation matrices
    print("Computing Affine Transformation.")
    affine_matrices = computeAffines(trianguations, caricature_vertices, mean_vertices)
    affine_matrices_inv = [linalg.inv(A) for A in affine_matrices]

    morph = warp(h, w, t, affine_matrices_inv, my_face)
    return morph

def makeMaleFemale():
    h = 480
    w = 640
    my_face = plt.imread("will_population.jpeg")/255

    mean_female_vertices = getAvgFemaleShape() # np array
    im_src_vertices = get_will_points() # np array
    scalar = .5
    caricature_vertices = scalar * (im_src_vertices - mean_female_vertices) + mean_female_vertices

    print("Computing Delaunay triangulation.")
    t = Delaunay(mean_female_vertices)
    trianguations = t.simplices

    # Compute Affine Transformation matrices
    print("Computing Affine Transformation.")
    affine_matrices = computeAffines(trianguations, caricature_vertices, mean_female_vertices)
    affine_matrices_inv = [linalg.inv(A) for A in affine_matrices]

    morph = warp(h, w, t, affine_matrices_inv, my_face)
    return morph


if __name__ == "__main__":
    face0 = plt.imread("will.jpg")
    face1 = plt.imread("emilie.jpg")
    assert(face0.shape == face1.shape), "Different size images. {} != {}".format(face0.shape, face1.shape)
    tri0_vertices_raw, tri1_vertices_raw = getPoints(face0, face1)
    assert(len(tri0_vertices_raw) == len(tri1_vertices_raw)), "Different amount of points selected. {} != {}".format(len(tri0_vertices_raw), len(tri1_vertices_raw))
    # print(tri0_vertices_raw)
    # print("~~~")
    # print(tri1_vertices_raw)
    # sys.exit(0)
    # Hardcoded Points

    assert(len(tri0_vertices_raw) == len(tri1_vertices_raw)), "Different amount of points selected. {} != {}".format(len(tri0_vertices_raw), len(tri1_vertices_raw))

    tri0_vertices = ginput_to_array(appendCorners(face0, tri0_vertices_raw))
    tri1_vertices = ginput_to_array(appendCorners(face1, tri1_vertices_raw))

    # Create Midway Face
    mid = createMidway(face0, face1, face0.shape[0], face0.shape[1], tri0_vertices, tri1_vertices)

    # Output Midway Face
    im_out_path = "midway_will_emilie2.png"
    plt.imsave(im_out_path, mid)
    # showImg(mid)

    # Create Warp
    frames = morph(face0, face1, face0.shape[0], face0.shape[1], tri0_vertices, tri1_vertices)
    save_gif(frames)
    print("Finished with creating gif.")

    # Calculate Mean Face of a population
    mean_face = population_text_parser()
    plt.imsave("mean_danish_face_with_me.png", mean_face)

    # Make Caricatures
    c0 = makeCaricature(-.5)
    c1 = makeCaricature(.5)
    plt.imsave("caricature0_1.png", c0)
    plt.imsave("caricature1_1.png", c1)

    # Bells and WhistleAttributeError
    bw = makeMaleFemale()
    plt.imsave("bells_and_whistle.png", bw)
