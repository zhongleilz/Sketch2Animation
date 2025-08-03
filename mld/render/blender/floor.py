import bpy
from .materials import floor_mat


def get_trajectory(data, is_mesh):
    if is_mesh:
        # mean of the vertices
        trajectory = data[:, :, [0, 1]].mean(1)
    else:
        # get the root joint
        trajectory = data[:, 0, [0, 1]]
    return trajectory


def plot_floor(data, big_plane=True):
    # Create a floor
    minx, miny, _ = data.min(axis=(0, 1))
    maxx, maxy, _ = data.max(axis=(0, 1))
    minz = 0

    location = ((maxx + minx)/2, (maxy + miny)/2, 0)
    # a little bit bigger
    scale = (1.08*(maxx - minx)/2, 1.08*(maxy - miny)/2, 1)

    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

    bpy.ops.transform.resize(value=scale, orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                             constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                             proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                             use_proportional_projected=False, release_confirm=True)
    obj = bpy.data.objects["Plane"]
    obj.name = "SmallPlane"
    obj.data.name = "SmallPlane"

    if not big_plane:
        obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
    else:
        obj.active_material = floor_mat(color=(0.1, 0.1, 0.1, 1))

    if big_plane:
        location = ((maxx + minx)/2, (maxy + miny)/2, -0.01)
        bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))

        bpy.ops.transform.resize(value=[2*x for x in scale], orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                 constraint_axis=(False, True, False), mirror=True, use_proportional_edit=False,
                                 proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                                 use_proportional_projected=False, release_confirm=True)

        obj = bpy.data.objects["Plane"]
        obj.name = "BigPlane"
        obj.data.name = "BigPlane"
        # obj.active_material = floor_mat(color=(0.2, 0.2, 0.2, 1))
        obj.active_material = floor_mat(color=(0.0, 0.0, 0.0, 1))


# def show_traj(coords):
#     """
#     使用 Bézier 曲线优化展示运动根轨迹，并减细线条。
#     参数：
#         coords (list of tuples): 包含 (x, y, z) 坐标的列表，表示轨迹点。
#     """
#     import bpy

#     # 创建曲线数据块
#     curve_data = bpy.data.curves.new('TrajectoryCurve', type='CURVE')
#     curve_data.dimensions = '3D'  # 设置为 3D
#     curve_data.resolution_u = 12  # 提高曲线的分辨率以显示平滑效果

#     # 创建 Bézier 样条
#     spline = curve_data.splines.new('BEZIER')  # 使用 Bézier 曲线
#     spline.bezier_points.add(len(coords) - 1)  # 添加点（默认已有1个点）

#     # 设置 Bézier 曲线的控制点
#     for i, coord in enumerate(coords):
#         x, y, z = coord  # 假设 coords 包含 (x, y, z)
#         point = spline.bezier_points[i]
#         point.co = (x, y, z)  # 设置控制点的位置
#         point.handle_left_type = 'AUTO'  # 左侧控制柄自动
#         point.handle_right_type = 'AUTO'  # 右侧控制柄自动

#     # 创建曲线对象
#     curve_object = bpy.data.objects.new('TrajectoryCurve', curve_data)
#     curve_data.bevel_depth = 0.002  # 减细线条厚度
#     curve_data.bevel_resolution = 0  # 不添加过多的剖面分辨率，确保细线条

#     # 将曲线添加到当前场景中
#     bpy.context.collection.objects.link(curve_object)

def show_traj(coords):
    import bpy

    # 创建曲线数据块
    curve_data = bpy.data.curves.new('TrajectoryCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12

    # 创建 Bézier 样条
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(coords) - 1)
    for i, coord in enumerate(coords):
        x, y, z = coord
        point = spline.bezier_points[i]
        point.co = (x, y, z)
        point.handle_left_type = 'AUTO'
        point.handle_right_type = 'AUTO'

    # 创建曲线对象
    curve_object = bpy.data.objects.new('TrajectoryCurve', curve_data)
    curve_data.bevel_depth = 0.002
    curve_data.bevel_resolution = 4

    # 创建材质
    mat = bpy.data.materials.new(name="GrayMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.2, 0.2, 0.2, 1.0)  # 浅灰色
        bsdf.inputs["Roughness"].default_value = 0.7
        bsdf.inputs["Specular"].default_value = 0.0

    # 应用材质
    curve_object.data.materials.append(mat)

    # 添加曲线到场景
    bpy.context.collection.objects.link(curve_object)




