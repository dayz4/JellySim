import numpy as np
from vec3 import Vec3


class MeshInfo:
    def __init__(self, vertices, normals, tetrahedrons, triangles, line_segments, line_vertices, radial_segments, radial_vertices):
        self.vertices = np.array(vertices, dtype="float32")
        self.normals = self.build_normals(normals)
        self.tetrahedrons = self.build_tetrahedrons(tetrahedrons)
        self.triangles = self.build_triangles(triangles)
        self.line_segments = self.build_line_segments(line_segments, np.array(line_vertices, dtype="float32"))
        self.radial_segments = self.build_line_segments(radial_segments, np.array(radial_vertices, dtype="float32"))

    @staticmethod
    def build_normals(normals_info):
        normals = []
        for x, y, z in normals_info:
            normals.append(Vec3(x, y, z))
        return normals

    def build_tetrahedrons(self, tetrahedrons_info):
        tetrahedrons = []
        for v1, v2, v3, v4 in tetrahedrons_info:
            vertices = [self.vertices[v1], self.vertices[v2], self.vertices[v3], self.vertices[v4]]
            tetrahedrons.append(Tetrahedron(vertices, [v1, v2, v3, v4]))
        return tetrahedrons

    @staticmethod
    def build_triangles(triangles_info):
        return np.array(triangles_info, dtype="uint32")

    @staticmethod
    def build_line_segments(line_segments_info, line_vertices):
        line_segments = []
        for v1, v2 in line_segments_info:
            endpoints = [line_vertices[v1], line_vertices[v2]]
            line_segments.append(LineSegment(endpoints, [v1, v2]))
        return line_segments


class Tetrahedron:
    def __init__(self, vertices, vertex_ids):
        self.p1 = vertices[0]
        self.p2 = vertices[1]
        self.p3 = vertices[2]
        self.p4 = vertices[3]
        self.vertex_ids = vertex_ids
        self.activation = 0.0


class LineSegment:
    def __init__(self, endpoints, vertex_ids):
        self.p1 = endpoints[0]
        self.p2 = endpoints[1]
        self.vertex_ids = vertex_ids
