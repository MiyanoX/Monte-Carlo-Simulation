from Class_def import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

x_multiple = 6


# create a Data class to acquire the data
class Data:
    def __init__(self, data_index):
        simulation_data = np.load("data/Ctt_data_{}.npy".format(data_index), allow_pickle=True)
        self.node_x = simulation_data[0]
        self.node_y = simulation_data[2]
        self.node_theta = simulation_data[3]
        self.layer_num = len(self.node_x)


# return the round value of y from point
def round_y(p: Point):
    return int(round((p.y - 0.5 * tape_thickness)/tape_thickness))


# change theta from 0 ~ 180 to 0 ~ 90
def theta_change(theta):
    if theta > pi/2:
        return pi - theta
    else:
        return theta


# create a Tape class to represent the tape object
class Tape:
    def __init__(self, start: Point, end: Point, theta=0, label=0):
        start_layer = round_y(start)
        end_layer = round_y(end)
        if start_layer != end_layer:
            raise TypeError('Wrong Two Points')
        self.start = start.x
        self.end = end.x
        self.layer_no = start_layer
        self.length = self.start - self.end
        self.label = label
        self.theta = theta_change(theta)
        self.mid = (self.start + self.end)/2

    def __repr__(self):
        return "Tape {0}: {1} -> {2} in layer {3}".format(self.label, self.start, self.end, self.layer_no)

    # return the start point of tape
    @property
    def start_point(self) -> Point:
        return Point(self.start, self.layer_no)

    # return the end point of tape
    @property
    def end_point(self) -> Point:
        return Point(self.end, self.layer_no)

    # return line segment
    @property
    def line(self):
        return LineSegment(self.start_point, self.end_point)

    # draw tape in Opencv
    def draw(self, point_color=(0, 0, 255), thickness=1, image=img):
        ptStart = (int(round(self.start * x_multiple)), self.layer_no * 2 * (thickness - 1) + thickness)
        ptEnd = (int(round(self.end * x_multiple)), self.layer_no * 2 * (thickness - 1) + thickness)
        lineType = 4
        cv2.line(image, ptStart, ptEnd, point_color, thickness, lineType)


# judge whether two tapes are adjacent
def if_adjacent(t1: Tape, t2: Tape):
    if abs(t1.layer_no - t2.layer_no) == 1:
        return True
    else:
        return False


# return the overlap length of two tapes
def overlap_length(t1: Tape, t2: Tape):
    if t1.start >= t2.end or t1.end <= t2.start:
        return 0
    elif t1.start < t2.start:
        if t1.end > t2.end:
            return t2.length
        else:
            return t1.end - t2.start
    else:
        if t2.end > t1.end:
            return t1.length
        else:
            return t2.end - t1.start


# judge if x of point is in range of tape
def if_point_in_tape(p: Point, tape: Tape):
    if tape.start < p.x < tape.end:
        return True
    else:
        return False


# create a sidelayer class to represent the a layer from the slice of specimen
class SideLayer:
    def __init__(self, layer_no):
        self.layer_no = layer_no
        self.tapes = []

    # add tape to sidelayer object
    def tape_add(self, tape):
        self.tapes.append(tape)

    # return the number of tapes in sidelayer
    @property
    def tape_number(self):
        return len(self.tapes)

    # draw the sidelayer object in Opencv
    def draw(self, point_color=(0, 0, 255), thickness=1, image=img):
        for tape in self.tapes:
            tape.draw(point_color, thickness, image)


# create a Side specimen class to represent the slice of specimen
class SideSpecimen:
    def __init__(self, layer_no):
        self.layers = []
        for i in range(layer_no):
            self.layers.append(SideLayer(i))

    # return a layer in side specimen according to its number
    def select_layer(self, layer_no):
        for layer in self.layers:
            if layer.layer_no == layer_no:
                return layer
        return None

    # return the number of layers in side specimen
    @property
    def layer_number(self):
        return len(self.layers)

    # return the number of tapes in specimen
    @property
    def tape_number(self):
        sum = 0
        for layer in self.layers:
            sum += layer.tape_number
        return sum

    # draw the side specimen in Opencv
    def draw(self, point_color=(0, 0, 255), thickness=1, image=img):
        for layer in self.layers:
            layer.draw(point_color, thickness, image)


# judge whether this tape can be the destruction path from point
def if_path_possible(p: Point, tape_p: Point, speci: SideSpecimen):
    for i in range(p.y + 1, tape_p.y):
        layer = speci.select_layer(i)
        for inter_tape in layer.tapes:
            line_tape = inter_tape.line
            line = LineSegment(p, tape_p)
            if line.intersect(line_tape):
                return False
    return True


# return the smallest distance from point to tape and return this point
def point_to_tape(p: Point, tape: Tape, speci: SideSpecimen):
    start_distance = abs(p.x - tape.start)
    end_distance = abs(p.x - tape.end)
    if start_distance < end_distance:
        if not if_path_possible(p, tape.start_point, speci):
            return float('inf'), None
        else:
            distance = (start_distance ** 2 + ((p.y - tape.layer_no) * tape_thickness) ** 2) ** 0.5
            return distance, tape.start_point
    else:
        if not if_path_possible(p, tape.end_point, speci):
            return float('inf'), None
        else:
            distance = (end_distance ** 2 + ((p.y - tape.layer_no) * tape_thickness) ** 2) ** 0.5
            return distance, tape.end_point


# use the result from data class to create a side specimen
def cal(data: Data):
    x = data.node_x
    y = data.node_y
    theta = data.node_theta
    layer_num = data.layer_num
    specimen = SideSpecimen(layer_num)
    label = 0

    for i in range(layer_num):
        tapeNo = len(x[i])
        for j in range(tapeNo):
            if theta[i][j] == 0:
                continue
            else:
                start = Point(x[i][j], y[i][j])
                if j + 1 < tapeNo:
                    end = Point(x[i][j+1], y[i][j+1])
                else:
                    end = Point(specimen_length, y[i][j])
                layer_no = round_y(end)
                tape = Tape(start, end, theta[i][j], label)
                label += 1
                layer = specimen.select_layer(layer_no)
                layer.tape_add(tape)

    return specimen


# return the nearest point from chosen tape and its tape
def nearest_point(p: Point, start_layer_no, speci: SideSpecimen, add):
    smallest_distance = float('inf')
    r1, r2 = None, None
    if start_layer_no + add <= speci.layer_number:
        end = start_layer_no + add
    else:
        end = speci.layer_number
    for i in range(start_layer_no, end):
        layer = speci.select_layer(i)
        for tape in layer.tapes:
            distance, point = point_to_tape(p, tape, speci)
            # print(distance, point)
            if distance < smallest_distance:
                smallest_distance = distance
                r1, r2 = tape, point
            else:
                continue
    return r1, r2


# Search Weakest Area by CLT
def c11_calculation(speci_tape: Tape, speci: SideSpecimen):
    # rule of mixture
    E1 = Ef * Vf + Em * (1 - Vf)
    E2 = 1 / (Vf / Ef + (1 - Vf) / Em)
    v12 = vf * Vf + vm * (1 - Vf)
    v21 = v12 * E2 / E1

    C11 = E1 / (1 - v12 * v21)
    C22 = E2 / (1 - v12 * v21)
    C12 = v12 * E2 / (1 - v12 * v21)
    C66 = G12

    C11_Specimen = 0
    C12_Specimen = 0
    C22_Specimen = 0
    C66_Specimen = 0
    C16_Specimen = 0
    C26_Specimen = 0

    # Calculate CLT
    for layer in speci.layers:

        temp_theta = 0
        tapes = layer.tapes

        for tape in tapes:
            overlap = overlap_length(tape, speci_tape)
            if overlap > 0:
                temp_theta = tape.theta
                break
            else:
                continue

        l = np.cos(temp_theta)
        m = np.sin(temp_theta)
        C11_Specimen += l ** 4 * C11 + 2 * l ** 2 * m ** 2 * (C12 + 2 * C66) + m ** 4 * C22
        C12_Specimen += l ** 2 * m ** 2 * (C11 + C12 - 4 * C66) + (l ** 4 + m ** 4) * C12
        C22_Specimen += m ** 4 * C11 + 2 * l ** 2 * m ** 2 * (C12 + 2 * C66) + l ** 4 * C22
        C66_Specimen += l ** 2 * m ** 2 * (C11 + C22 - 2 * C12) + (l ** 2 - m ** 2) * C66
        C16_Specimen += l ** 3 * m * (2 * C66 - C11 + C12) - l * m ** 3 * (2 * C66 - C22 + C12)
        C26_Specimen += l * m ** 3 * (2 * C66 - C11 + C12) - l ** 3 * m * (2 * C66 - C22 + C12)

    return C11_Specimen / speci.layer_number


# search the smallest in clt
def clt_search(speci: SideSpecimen, div=specimen_length):
    clt = []
    for i in range(1, div-1):
        speci_tape = Tape(Point(i,0), Point(i+1,0))
        temp = c11_calculation(speci_tape, speci)
        # print(temp)
        clt.append(temp)
    return clt.index(min(clt)) + 0.5


# return the weakest tape
def clt_node(speci: SideSpecimen, sp):
    clt_res = sp
    # print(clt_res)
    tapes = speci.select_layer(0).tapes
    min_tape = tapes[0]
    min_point = min_tape.start_point
    ###########
    # !!!!!!! #
    ###########
    min_dist = abs(min_tape.mid - clt_res)
    ###########
    # !!!!!!! #
    ###########
    for tape in speci.select_layer(0).tapes:
        start_distance = abs(sp - tape.start)
        end_distance = abs(sp - tape.end)
        if start_distance < end_distance:
            distance = start_distance
            point = tape.start_point
        else:
            distance = end_distance
            point = tape.end_point
        if distance < min_dist:
            min_tape = tape
            min_dist = distance
            min_point = point
        else:
            continue
    return min_point, min_tape


# calculation the start point
def clt_point(div=specimen_length):
    data = Data(1)
    thick = 4
    s = cal(data)
    return clt_search(s, div), s.layer_number


"""
FRACTURE STRENGTH FOR ONE TAPE BASED ON HASHIN CRITERION.
"""


def s13_to_length(theta):

    mode_interface = np.deg2rad(15.)  # 15 degrees from Hashin's paper

    if theta <= mode_interface:
        stress = (1 / (((np.cos(theta)) ** 2 / S11_plus) ** 2 + (np.sin(theta) * np.cos(theta) / S12) ** 2)) ** 0.5
        res = stress * tape_thickness / S12

    elif theta > mode_interface:
        stress = (1 / (((np.sin(theta)) ** 2 / S22_plus) ** 2 + (np.sin(theta) * np.cos(theta) / S12) ** 2)) ** 0.5
        res = stress * tape_thickness / S12

    else:
        print('Error in Hashin criterion')

    return res


def path_finding(p: Point, prev_tape: Tape, speci: SideSpecimen, add=3):
    if prev_tape.layer_no == speci.layer_number - 1:
        return [p], 0

    overlap_tape = None
    next_layer = speci.select_layer(prev_tape.layer_no + 1)
    for tape in next_layer.tapes:
        if if_point_in_tape(p, tape):
            overlap_tape = tape
            break
        else:
            continue
    # print(p, overlap_tape)
    if overlap_tape:
        length = overlap_length(prev_tape, overlap_tape)
        if length >= L_lim:
            new_point = Point(p.x, next_layer.layer_no)
            path, fracture_tape_num = path_finding(new_point, overlap_tape, speci)
            return [p] + path, fracture_tape_num + 1
        else:
            start_distance = abs(p.x - overlap_tape.start)
            end_distance = abs(p.x - overlap_tape.end)
            if start_distance < end_distance:
                new_point = overlap_tape.start_point
            else:
                new_point = overlap_tape.end_point
            path, fracture_tape_num = path_finding(new_point, overlap_tape, speci)
            return [p] + path, fracture_tape_num

    new_tape, new_point = nearest_point(p, next_layer.layer_no, speci, add)
    # print(' -> ', new_tape, new_point)
    path, fracture_tape_num = path_finding(new_point, new_tape, speci)
    return [p] + path, fracture_tape_num


# draw the path of destruction in Opencv
def path_draw(path, point_color=(0, 0, 255), thickness=1, image=img):
    for i in range(len(path) - 1):
        ptStart = (int(round(path[i].x * x_multiple)), path[i].y * 2 * (thickness - 1) + thickness)
        ptEnd = (int(round(path[i+1].x * x_multiple)), path[i+1].y * 2 * (thickness - 1) + thickness)
        lineType = 4
        cv2.line(image, ptStart, ptEnd, point_color, thickness, lineType)


# return the distance from two points in path
def path_point_distance(p1: Point, p2: Point):
    x_2 = (p1.x - p2.x) ** 2
    y_2 = ((p1.y - p2.y) * tape_thickness) ** 2
    return (x_2 + y_2) ** 0.5


# return the length of whole path
def path_length(path):
    length = 0
    for i in range(len(path) - 1):
        length += path_point_distance(path[i], path[i+1])
    return length


# change image_i's alpha
def alpha_change(i, alpha=30):
    img = cv2.imread("image/path_{}.png".format(i))

    b_channel, g_channel, r_channel = cv2.split(img)

    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    cv2.imwrite("alpha_30/path_{}.png".format(i), img_BGRA)

    return img_BGRA


# calculation the path of in this side specimen
def path_calculation(i, image, sp):
    data = Data(i)
    thick = 4
    s = cal(data)
    start_point, start_tape = clt_node(s, sp)  # start tape
    s.draw(point_color=(0, 0, 0), thickness=thick, image=image)
    start_tape.draw(point_color=(0, 255, 0), thickness=thick, image=image)
    path, fracture_tape_num = path_finding(start_point, start_tape, s, add_layers)
    path_draw(path, point_color=(0, 0, 255), thickness=thick, image=image)
    length = path_length(path)
    print("path's length = ", length)
    # draw
    # cv2.namedWindow("image")
    # cv2.imshow('image', image)
    # cv2.waitKey(100)
    cv2.imwrite('image/path_{}.png'.format(i), image)
    alpha_change(i)
    cv2.destroyAllWindows()
    return fracture_tape_num, [path, i]


def change_2D_to_3D(points):
    res = []
    for point in points[0]:
        res.append([point.x, point.y, points[1]])
    return res


# calculation the strength of side specimen according to the destruction path
def strength_calculation(i, image, sp):
    fracture_tape_num, path_points = path_calculation(i, image, sp)
    # Calculate the stress
    stress_sum = (fracture_tape_num * S12)
    print('stress =', stress_sum, 'MPa')
    return stress_sum, change_2D_to_3D(path_points)


def cube(ax, thickness):
    verts = [(0, 0, 0), (0, specimen_width, 0), (specimen_length, specimen_width, 0), (specimen_length, 0, 0),
             (0, 0, thickness), (0, specimen_width, thickness), (specimen_length, specimen_width, thickness), (specimen_length, 0, thickness)]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
    # 四面体顶点和面
    # verts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)]
    # faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    # 获得每个面的顶点
    poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
    # print(poly3d)

    # 绘制顶点
    x, y, z = zip(*verts)
    ax.scatter(x, y, z)
    # 绘制多边形面
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=1, alpha=0.3))
    # 绘制对变形的边
    ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.5, linestyles=':'))


def scatter_3d(ax, points):
    x, y, z = [], [], []
    for point in points:
        x.append(point[0])
        y.append(point[2])
        z.append(point[1])

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    ax.scatter(x, y, z, s=1)

    # X, Y, Z = np.meshgrid(x, y, z)
    # ax.plot_surface(X, Y, Z)


# A demo to analyze every slice in specimen
def demo(num):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    strength_sum = 0
    sp, thickness = clt_point()
    path_point_3d = []
    for i in range(1, 1 + num):
        print('\ndata', i, 'start...')
        image = np.ones((specimen_width * x_multiple, specimen_length * x_multiple, 3), np.uint8) * 255
        i_sum, i_point = strength_calculation(i, image, sp)
        strength_sum += i_sum
        path_point_3d += i_point
    avg = strength_sum / num
    print("specimen's average stress is", avg, 'MPa')

    # cube(ax, thickness)
    # scatter_3d(ax, path_point_3d)
    # ax.set_xlabel('X')
    # ax.set_xlim3d(0, 300)
    # ax.set_ylabel('Y')
    # ax.set_ylim3d(0, 300)
    # ax.set_zlabel('Z')
    # ax.set_zlim3d(0, 200)
    # plt.show()

    return avg


# demo(125)


