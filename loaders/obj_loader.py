from muscle import Muscle


def load(fn):
    f = open(fn + ".obj", "r")
    file_lines = f.readlines()
    vertices, normals, faces, line_segments = [], [], [], []
    for file_line in file_lines:
        file_line = file_line.rstrip()
        if file_line == '':
            continue

        identifier, coords = file_line.split(" ", 1)

        if identifier == 'v':
            vertex = parse_vertex(coords)
            vertices.append(vertex)

        if identifier == 'vn':
            normal = parse_vertex(coords)
            normals.append(normal)

        if identifier == 'f':
            face = parse_face(coords)
            faces.append(face)

        if identifier == 'l':
            line_segment = parse_line_segment(coords)
            line_segments.append(line_segment)

    return line_segments, vertices


def parse_vertex(coords):
    coords_list = coords.split()
    return [float(coords_list[0]), -float(coords_list[2]), float(coords_list[1])]


def parse_face(vertices):
    vertices_list = vertices.split()
    face_vertices = []
    for vertices in vertices_list:
        # [v, vn] = vertices.split("//")
        # face_vertices.append(int(v)-1)
        # print(vertices)
        face_vertices.append(int(vertices)-1)
    return face_vertices


def parse_line_segment(vertices):
    vertex_list = vertices.split()
    line_vertices = []
    for vertex in vertex_list:
        line_vertices.append(int(vertex)-1)
    return line_vertices

