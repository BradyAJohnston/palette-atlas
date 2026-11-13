"""Evaluate the built library's node groups in each editor, for verify.py.

Each case gives a palette, its inputs, and columns: column c reads Value
(c + offset) / scale, and Index c. Geometry Nodes stores the colour on points;
Shader and Compositor render one pixel per column through the Standard view
transform, with the colour going straight to the output, so nothing but the
node group touches it. The results are display sRGB, 0 to 1.
"""

import tempfile
from pathlib import Path

import bpy
import numpy as np

LIBRARY_DIR = Path(__file__).parents[2] / "library"
EDITORS = {"geometry": "GeometryNodeGroup", "shader": "ShaderNodeGroup", "compositor": "CompositorNodeGroup"}
HEIGHT = 4


def to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def load_group(editor: str, collection: str, name: str) -> bpy.types.NodeTree:
    path = LIBRARY_DIR / editor / f"{collection}.blend"
    with bpy.data.libraries.load(str(path), link=False) as (src, dst):
        dst.node_groups = [name]
    return dst.node_groups[0]


def add(tree, idname, **props):
    node = tree.nodes.new(idname)
    for k, v in props.items():
        setattr(node, k, v)
    return node


def math(tree, op, a, b):
    node = add(tree, "ShaderNodeMath", operation=op)
    for socket, v in zip(node.inputs, (a, b)):
        if isinstance(v, bpy.types.NodeSocket):
            tree.links.new(v, socket)
        else:
            socket.default_value = v
    return node.outputs[0]


def drive_group(tree, group, column, case):
    """A group node reading Value and Index from the column number."""
    node = add(tree, EDITORS[case["editor"]], node_tree=group)
    value = math(tree, "DIVIDE", math(tree, "ADD", column, case["offset"]), case["scale"])
    tree.links.new(value, node.inputs["Value"])
    if "Index" in node.inputs:
        tree.links.new(column, node.inputs["Index"])
    node.inputs["Reverse"].default_value = case["reverse"]
    if case["mode"]:
        # a group's menu only lists its items once the host tree is updated
        tree.interface_update(bpy.context)
        bpy.context.evaluated_depsgraph_get().update()
        node.inputs["Mode"].default_value = case["mode"]
    return node.outputs["Color"]


def eval_geometry(group, case):
    count = case["count"]
    mesh = bpy.data.meshes.new("points")
    mesh.vertices.add(count)
    obj = bpy.data.objects.new("points", mesh)
    bpy.context.scene.collection.objects.link(obj)
    tree = bpy.data.node_groups.new("verify", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin, gout = add(tree, "NodeGroupInput"), add(tree, "NodeGroupOutput")
    column = add(tree, "GeometryNodeInputIndex").outputs[0]
    store = add(tree, "GeometryNodeStoreNamedAttribute", data_type="FLOAT_COLOR", domain="POINT")
    store.inputs["Name"].default_value = "color"
    tree.links.new(gin.outputs[0], store.inputs["Geometry"])
    tree.links.new(drive_group(tree, group, column, case), store.inputs["Value"])
    tree.links.new(store.outputs[0], gout.inputs[0])
    obj.modifiers.new("verify", "NODES").node_group = tree

    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    data = np.zeros(count * 4, np.float32)
    evaluated.data.attributes["color"].data.foreach_get("color", data)
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)
    bpy.data.node_groups.remove(tree)
    return to_srgb(data.reshape(-1, 4)[:, :3].astype(float))


def setup_render(scene, width, path):
    scene.render.resolution_x, scene.render.resolution_y = width, HEIGHT
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.display_settings.display_device = "sRGB"
    scene.render.dither_intensity = 0.0
    settings = scene.render.image_settings
    settings.file_format, settings.color_depth, settings.color_mode = "PNG", "16", "RGB"
    scene.render.filepath = str(path)


def render(scene, path, width):
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(path))
    image.colorspace_settings.name = "Non-Color"  # the display values as written
    pixels = np.array(image.pixels[:]).reshape(HEIGHT, width, 4)
    bpy.data.images.remove(image)
    return pixels[0, :, :3]


def pixel_column(tree, coordinate, width):
    sep = add(tree, "ShaderNodeSeparateXYZ")
    tree.links.new(coordinate, sep.inputs[0])
    return math(tree, "FLOOR", math(tree, "MULTIPLY", sep.outputs["X"], width), 0.0)


def setup_shader_scene(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01  # each pixel samples its own centre
    scene.cycles.max_bounces = 0
    scene.world = bpy.data.worlds.new("black")
    scene.world.color = (0, 0, 0)
    if scene.world.node_tree:
        scene.world.node_tree.nodes.clear()
    mesh = bpy.data.meshes.new("plane")
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("plane", mesh)
    scene.collection.objects.link(plane)
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type, camera.data.ortho_scale = "ORTHO", 1.0
    camera.location = (0, 0, 2)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return plane


def eval_shader(group, case, plane, path):
    scene = bpy.context.scene
    setup_render(scene, case["count"], path)
    material = bpy.data.materials.new("verify")
    tree = material.node_tree
    tree.nodes.clear()
    window = add(tree, "ShaderNodeTexCoord").outputs["Window"]
    color = drive_group(tree, group, pixel_column(tree, window, case["count"]), case)
    out = add(tree, "ShaderNodeOutputMaterial")
    tree.links.new(color, out.inputs["Surface"])  # the colour as emission, unlit
    plane.data.materials.clear()
    plane.data.materials.append(material)
    pixels = render(scene, path, case["count"])
    bpy.data.materials.remove(material)
    return pixels


def eval_compositor(group, case, path):
    scene = bpy.context.scene
    width = case["count"]
    setup_render(scene, width, path)
    tree = bpy.data.node_groups.new("verify", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    blank = add(tree, "CompositorNodeBlankImage")
    blank.inputs["Size"].default_value = (width, HEIGHT)
    coordinates = add(tree, "CompositorNodeImageCoordinates")
    tree.links.new(blank.outputs[0], coordinates.inputs[0])
    color = drive_group(tree, group, pixel_column(tree, coordinates.outputs["Normalized"], width), case)
    tree.links.new(color, add(tree, "NodeGroupOutput").inputs[0])
    pixels = render(scene, path, width)
    scene.compositing_node_group = None
    bpy.data.node_groups.remove(tree)
    return pixels


def evaluate(cases: list[dict]) -> dict[str, np.ndarray]:
    """Display sRGB per case, keyed "<editor>/<case index>"."""
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "render.png"
        for editor in EDITORS:
            bpy.ops.wm.read_homefile(use_factory_startup=True, use_empty=True, load_ui=False)
            scene = bpy.context.scene
            scene.render.use_compositing = editor == "compositor"
            plane = setup_shader_scene(scene) if editor == "shader" else None
            if editor == "compositor":
                scene.render.engine = "CYCLES"
            groups = {}
            for i, case in enumerate(cases):
                case = {**case, "editor": editor}
                key = (case["collection"], case["name"])
                if key not in groups:
                    groups[key] = load_group(editor, *key)
                if editor == "geometry":
                    results[f"{editor}/{i}"] = eval_geometry(groups[key], case)
                elif editor == "shader":
                    results[f"{editor}/{i}"] = eval_shader(groups[key], case, plane, png)
                else:
                    results[f"{editor}/{i}"] = eval_compositor(groups[key], case, png)
            print(f"evaluated {editor}: {len(cases)} cases", flush=True)
    return results
