import mesh


def load(fn):
    f = open(fn + ".vtk", "r")
    lines = f.readlines()
    vertices, normals, tetrahedrons, triangles = [], [], [], []
    read_vertices, read_tetrahedrons = False, False
    for line in lines:
        line = line.rstrip()
        if line == '':
            continue
        elif line == '# vertices':
            read_vertices = True
            read_tetrahedrons = False
        elif line == '# tetrahedrons':
            read_tetrahedrons = True
            read_vertices = False
        else:
            if read_vertices:
                vertex = parse_vertex(line)
                vertices.append(vertex)
            elif read_tetrahedrons:
                tetrahedron, tetrahedron_triangles = parse_tetrahedron(line)
                tetrahedrons.append(tetrahedron)
                triangles.extend(tetrahedron_triangles)

    return vertices, normals, tetrahedrons, triangles


def parse_vertex(coords):
    coords_list = coords.split()
    return [float(coords_list[0]), float(coords_list[1]), float(coords_list[2])]


def parse_tetrahedron(line):
    tetrahedron = line.split()[1:]
    v0, v1, v2, v3 = [int(v) for v in tetrahedron]
    triangles = [[v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]]
    return [v0, v1, v2, v3], triangles
