import math
from numpy import sin, cos, pi
import numpy as np
import scipy.spatial as spatial
import cv2


#############
# Parameter #
#############


# size of specimen (JIS)5
specimen_thickness = 2  # 2
specimen_length = 250
specimen_width = 125

# size and number of tape
tape_thickness = 0.044
tape_length = 18
tape_width = 5
Ave_Stack = round(specimen_thickness / tape_thickness)
tape_area = tape_length * tape_width
layer_multiple = 1.5
void_theta = 0

# Import material information
Ef = 240                      # GPa
Em = 2.6                      # GPa
Vf = 0.5                      # Volume fraction
layer_vf = 0.5                # Volume fraction per layer
G12 = 3.0                     # GPa
vf = 0.24                     # Poisson ratio of fiber
vm = 0.38                     # Poisson ration of matrix


# import necessary data, from [Ref]
S11_plus = 2900  # MPa
S12 = 75  # MPa
S22_plus = 105  # MPa
L_lim = 1.77                  # mm; this data is from the [REF]


# simulation data
max_random_time = 1000
data_num = 125

# destruction
add_layers = 3

# draw
multiple = 8
img = np.ones((specimen_width * multiple, specimen_length * multiple,  3), np.uint8) * 255


##################
# Class & Method #
##################


''' define a Point class represent a point '''


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def coordinate(self):
        return (self.x, self.y)

    @property
    def coordinate_round(self):
        return (round(self.x), round(self.y))

    def __repr__(self):
        return "({0}, {1})".format(self.x, self.y)

    def slope_to(self, p):
        if self.x == p.x:
            return float("inf")
        return (self.y - p.y) / (self.x - p.x)

    def distance_to(self, p):
        return math.sqrt(abs(self.x * self.x - p.x * p.x) + abs(self.y * self.y - p.y * p.y))

    def another_point(self, length, angle):
        x = self.x + length * cos(angle)
        y = self.y + length * sin(angle)
        return Point(x, y)


''' define a Line_Segment class represent a line segment '''


def triangle_area(a: Point, b: Point, c: Point):
    return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)


class LineSegment:
    def __init__(self, p: Point, q: Point):
        self.p = p
        self.q = q
        self.slope = p.slope_to(self.q)

    def __repr__(self):
        return "Line: {0} -> {1}".format(self.p, self.q)

    @property
    def length(self):
        return self.p.slope_to(self.q)

    ''' judge if it is possilbe that two line segment intersect '''

    def intersect(self, line):
        if self.slope == line.slope:
            return False
        if max(self.p.x, self.q.x) < min(line.p.x, line.q.x) \
                or max(self.p.y, self.q.y) < min(line.p.y, line.q.y) \
                or max(line.p.x, line.q.x) < min(self.p.x, self.q.x) \
                or max(line.p.y, line.q.y) < min(self.p.y, self.q.y):
            return False
        return True

    ''' calculate intersect between two line segment '''
    ''' https://blog.csdn.net/u012260672/article/details/51941262?utm_medium=distribute.pc_rel
    evant.none-task-blog-baidujs-2'''

    def intersect_point(self, line):
        if not self.intersect(line):
            return None

        a, b, c, d = self.p, self.q, line.p, line.q

        area_abc = triangle_area(a, b, c)
        area_abd = triangle_area(a, b, d)
        if area_abc * area_abd >= 0:
            return None

        area_cda = triangle_area(c, d, a)
        area_cdb = triangle_area(c, d, b)
        if area_cda * area_cdb >= 0:
            return None

        t = area_cda / (area_abd - area_abc)
        dx, dy = t * (b.x - a.x), t * (b.y - a.y)
        x, y = a.x + dx, a.y + dy
        return Point(x, y)

    def draw(self, point_color=(0, 0, 255), thickness=1):  # color BRG
        ptStart = (int(round(self.p.x * multiple)), int(round(self.p.y * multiple)))
        ptEnd = (int(round(self.q.x * multiple)), int(round(self.q.y * multiple)))
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)


'''child class AxisLine include a property contains the angle data from the intersect rectangle'''


class AxisLine(LineSegment):
    def __init__(self, p: Point, q: Point, theta, z):
        self.theta = theta
        self.z = z
        if p.x == q.x:
            self.axis = 'x'
            if p.y < q.y:
                LineSegment.__init__(self, p, q)
            else:
                LineSegment.__init__(self, q, p)
        else:
            self.axis = 'y'
            if p.x < q.x:
                LineSegment.__init__(self, p, q)
            else:
                LineSegment.__init__(self, q, p)

    def __repr__(self):
        return "Axis" + LineSegment.__repr__(self)

    @property
    def min_coordinate(self):
        if self.axis == 'x':
            return self.p.y
        else:
            return self.p.x

    def __lt__(self, other):
        if not isinstance(other, AxisLine):
            raise TypeError('Not a AxisLine class')
        elif self.min_coordinate < other.min_coordinate:
            return True
        else:
            return False


''' define a Rectangle class represent a tape '''


class Rectangle:
    def __init__(self, p: Point, angle, length=tape_length, width=tape_width):

        x1 = + length / 2 * cos(angle) - width / 2 * sin(angle) + p.x
        y1 = + length / 2 * sin(angle) + width / 2 * cos(angle) + p.y
        x2 = - length / 2 * cos(angle) - width / 2 * sin(angle) + p.x
        y2 = - length / 2 * sin(angle) + width / 2 * cos(angle) + p.y
        x3 = - length / 2 * cos(angle) + width / 2 * sin(angle) + p.x
        y3 = - length / 2 * sin(angle) - width / 2 * cos(angle) + p.y
        x4 = + length / 2 * cos(angle) + width / 2 * sin(angle) + p.x
        y4 = + length / 2 * sin(angle) - width / 2 * cos(angle) + p.y

        p1 = Point(x1, y1)
        p2 = Point(x2, y2)
        p3 = Point(x3, y3)
        p4 = Point(x4, y4)

        self.center = p
        self.angle = angle
        self.points = [p1, p2, p3, p4]

    def __repr__(self):
        p = self.points
        return "Rectangle: [{0}, {1}, {2}, {3}]".format(p[0], p[1], p[2], p[3])

    @property
    def coordinate(self):
        return self.center.coordinate

    @property
    def lines(self):
        p = self.points
        return [LineSegment(p[0], p[1]), LineSegment(p[1], p[2]), LineSegment(p[2], p[3]), LineSegment(p[3], p[0])]

    def intersect(self, r):
        for i in self.lines:
            for j in r.lines:
                if i.intersect(j):
                    return True
        return False

    def axis_intersect(self, axis, value, z):
        if axis == 'x':
            line = LineSegment(Point(value, -10000), Point(value, 10000))
        elif axis == 'y':
            line = LineSegment(Point(-10000, value), Point(10000, value))
        else:
            raise AssertionError('Axis input error')
        points = []
        for l in self.lines:
            res = l.intersect_point(line)
            if res:
                points.append(res)
        if len(points) < 2:
            return None
        else:
            return AxisLine(points[0], points[1], self.angle, z)

    def draw(self, point_color=(0, 0, 255), thickness=1):
        for line in self.lines:
            line.draw(point_color, thickness)


'''define a Layer class to represent a layer in specimen'''


class Layer:
    def __init__(self, number, length, width, thickness):
        self.items = []
        self.points = []
        self.number = number
        self.length = length
        self.width = width
        self.thickness = thickness

    def __str__(self):
        return "Layer No.{0}: {1}mm * {2}mm * {3}mm with {4} rectangles".format(self.number, self.length, self.width, self.thickness, len(self.items))

    def add_item(self, x):
        self.items.append(x)
        self.points.append(x.coordinate)
        return True

    def add_item_judge(self, x):
        if len(self.points) == 0:
            return self.add_item(x)
        distance = math.sqrt(tape_length * tape_length + tape_width * tape_width)
        point_tree = spatial.cKDTree(self.points)
        index = point_tree.query_ball_point(x.coordinate, distance)
        for i in index:
            if x.intersect(self.items[i]):
                return False
        return self.add_item(x)

    def add_random_rectangle(self):
        # print(len(self.items))
        res = False
        i = 0
        while (not res) & (i < max_random_time):
            # print("  " + str(i))
            i += 1
            x = np.random.rand() * self.length
            y = np.random.rand() * self.width
            theta = np.random.rand() * pi
            res = self.add_item_judge(Rectangle(Point(x, y), theta))

    @property
    def vf(self):
        area = self.length * self.width
        return len(self.items) * tape_area / area

    def axis_intersect(self, axis, value):
        lines = []
        for item in self.items:
            res = item.axis_intersect(axis, value, self.number)
            if res:
                lines.append(res)
        lines.sort()
        return lines

    def draw(self, point_color=(0, 0, 0), thickness=1):
        for i in self.items:
            i.draw(point_color, thickness)


'''define a Specimen class to represent a specimen in simulation'''


class Specimen:
    def __init__(self, length=specimen_length, width=specimen_width, thickness=specimen_thickness, tape_thickness=tape_thickness):
        self.length = length
        self.width = width
        self.thickness = thickness
        self.layers = []
        self.layer_num = int(thickness/tape_thickness) + 1
        for i in range(int(self.layer_num * layer_multiple)):
            self.layers.append(Layer(i+1, length, width, tape_thickness))

    def __repr__(self):
        for layer in self.layers:
            print(layer)
        return "Specimen: {0}mm * {1}mm * {2}mm with {3} layers\nvf = {4}".format(self.length, self.width, self.thickness, len(self.layers), self.vf)

    def layer(self, number):
        return self.layers[number-1]

    @property
    def vf(self):
        vf_sum = 0
        for layer in self.layers:
            vf_sum += layer.vf
        return vf_sum / self.layer_num

    def add_random_tape(self):
        x = np.random.rand() * self.length
        y = np.random.rand() * self.width
        theta = np.random.rand() * pi
        i = 1
        res = False
        while (not res) & (i <= len(self.layers)):
            cur_layer = self.layer(i)
            if cur_layer.vf < layer_vf:
                res = cur_layer.add_item_judge(Rectangle(Point(x, y), theta))
            i += 1
        return res

    def axis_intersect(self, axis, value):
        layers = []
        for layer in self.layers:
            layers.append(layer.axis_intersect(axis, value))
        return layers

    def draw(self, layer):
        self.layer(layer).draw()


class AxisIntersect:
    def __init__(self, simulation, axis, value):
        self.layers = simulation.axis_intersect(axis, value)
        self.axis = axis
        self.value = value

    def __repr__(self):
        return "Axis {0} = {1} with {2} lines".format(self.axis, self.value, self.lines_num)

    @property
    def lines_num(self):
        i = 0
        for line in self.layers:
            for l in line:
                i += 1
        return i

    def layer_lines(self, layer_num):
        return self.layers[layer_num - 1]

    def draw(self):
        line = self.layers[0]
        for l in line:
            l.draw(point_color=(0, 0, 255))


class MonteCarlo:
    def __init__(self):
        self.sim = Specimen()
        self.axes = []
        j = 0
        while self.sim.vf < Vf:
            print(j)
            self.sim.add_random_tape()
            j += 1

    def __repr__(self):
        return self.sim.__repr__() + "\nwith {0} axes in {1} axis".format(self.axis_num, self.axis)

    @property
    def axis_num(self):
        return len(self.axes)

    @property
    def axis(self):
        if len(self.axes) == 0:
            return 'Wrong'
        else:
            return self.axes[0].axis

    def axis_intersect(self, axis, axis_num):
        if axis == 'x':
            length = specimen_length
        elif axis == 'y':
            length = specimen_width
        else:
            raise AssertionError('Axis input error')
        for i in range(axis_num):
            base = length / (axis_num + 1)
            # print(int((i + 1) * base))
            self.axes.append(AxisIntersect(self.sim, axis, int((i + 1) * base)))

    def data(self, axis_num):
        layers = self.axes[axis_num-1].layers
        x, y, z, theta = [], [], [], []
        for layer in layers:
            nord_x = []
            nord_theta = []
            for line in layer:
                nord_x.append(line.p.x)
                nord_x.append(line.q.x)
                nord_theta.append(line.theta)
                nord_theta.append(void_theta)
            if len(nord_x) > 1 and nord_x[1] < 0:
                nord_x = nord_x[2:]
            if nord_x[-2] > specimen_length:
                nord_x = nord_x[:-2]
            if nord_x[0] < 0:
                nord_x = [0] + nord_x[1:]
            else:
                nord_x = [0] + nord_x
                nord_theta = [void_theta] + nord_theta
            if nord_x[-1] > specimen_length:
                nord_x = nord_x[:-1]
                nord_theta = nord_theta[:-1]
            nord_y = np.ones(len(nord_x)) * layer[0].p.y
            nord_z = np.ones(len(nord_x)) * layer[0].z * tape_thickness - tape_thickness / 2
            x.append(np.array(nord_x))
            y.append(nord_y)
            z.append(nord_z)
            theta.append(np.array(nord_theta))
        a = np.array([np.array(x), np.array(y), np.array(z), np.array(theta)])
        return a

    def draw(self, layer):
        self.sim.draw(layer)
        # for axis in self.axes:
        #     axis.draw()

    # def sim_data(self):

#
# a = MonteCarlo()
# a.axis_intersect('y', 10)
#
#
# cv2.namedWindow("image")
# cv2.imshow('image', Ctt.img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()