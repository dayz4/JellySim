from loaders import vtk_loader, obj_loader, msh_loader
from mesh_info import MeshInfo
from mesh import Mesh
from muscle import Muscle
from jellyfish import Jellyfish
from fem_element import FEMElement
from radial_muscle import RadialMuscle


def load():
    # vertices, normals, tetrahedrons, triangles = vtk_loader.load("models/jellyfish3")
    vertices, normals, tetrahedrons, triangles = msh_loader.load("models/jellyfish4")
    line_segments, line_vertices = obj_loader.load("models/muscles4")
    radial_segments, radial_vertices = obj_loader.load("models/radial3")
    _, rhopalia = obj_loader.load("models/rhopalia")
    mesh_info = MeshInfo(vertices, normals, tetrahedrons, triangles, line_segments, line_vertices, radial_segments, radial_vertices)
    mesh = build_mesh(mesh_info)
    muscles = build_muscles(mesh_info)
    radial_muscles = build_radial(mesh_info)
    fem_elements = build_fem_elements(mesh_info)
    return Jellyfish(mesh, muscles, radial_muscles, fem_elements, rhopalia)


def build_mesh(mesh_info):
    return Mesh(mesh_info.vertices, mesh_info.normals, mesh_info.tetrahedrons, mesh_info.triangles)


def build_muscles(mesh_info):
    muscles = []
    for line_segment in mesh_info.line_segments:
        muscles.append(Muscle(line_segment))
    return muscles


def build_radial(mesh_info):
    radial = []
    for radial_segment in mesh_info.radial_segments:
        radial.append(RadialMuscle(radial_segment))
    return radial


def build_fem_elements(mesh_info):
    fem_elements = []
    for tetrahedron in mesh_info.tetrahedrons:
        fem_elements.append(
            FEMElement([tetrahedron.p1, tetrahedron.p2, tetrahedron.p3, tetrahedron.p4], tetrahedron.vertex_ids))
    return fem_elements
