import math
import os
import sys

import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, mesh_detect
from .vertices import prepare_vertices

from mathutils import Matrix

def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
    

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))


    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2] # # extract components back out of the matrix
   
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv@R_world2bcam
    T_world2cv = R_bcam2cv@T_world2bcam
#    print("R_world2cv is {}".format(R_world2cv))
#    print("T_world2cv is {}".format(T_world2cv))
#    print("R_world2cv[0][:] + (T_world2cv[0],) is {}".format(R_world2cv[0][:] + (T_world2cv[0], )))
    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K@RT, K, RT


def prune_begin_end(data, perc):
    to_remove = int(len(data)*perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


# def render_current_frame(path):
#     bpy.context.scene.render.filepath = path
#     bpy.ops.render.render(use_viewport=True, write_still=True)


# def render_current_frame(path):
#     # camera = bpy.data.objects['Camera']
#     # camera_matrix = camera.matrix_world
#     # camera_location = camera.location
#     # camera_rotation = camera.rotation_euler

#     # # 获取相机的投影矩阵
#     # projection_matrix = camera.calc_matrix_camera(
#     #     bpy.context.evaluated_depsgraph_get(),
#     #     x = bpy.context.scene.render.resolution_x,
#     #     y = bpy.context.scene.render.resolution_y
#     # )

#     # # print("projection_matrix",projection_matrix)
#     # projection_matrix = np.array(projection_matrix)

#     # point_3d = np.array([1.0, 2.0, 3.0])

#     # # 将 3D 点投影到图像平面上
#     # point_3d_homogeneous = np.append(point_3d, 1)  # 添加齐次坐标
#     # point_2d_homogeneous = projection_matrix @ camera_matrix @ point_3d_homogeneous

#     # point_2d_homogeneous = point_2d_homogeneous / point_2d_homogeneous[3]

#     # # 提取前两个分量，即图像平面上的 2D 坐标
#     # point_2d = point_2d_homogeneous[:2]

#     # print("point_2d",point_2d)

#     camera = bpy.data.objects['Camera']
#     P, K, RT = get_3x4_P_matrix_from_blender(camera)
#     P = np.array(P)
#     point_3d = np.array([1.0, 2.0, 3.0,1])
#     p1 = P @ point_3d
#     p1 /= p1[2]

#     print("P",P)
#     print("K",K)
#     print("RT",RT)

#     print("p1",p1)
#     #####
#     # # 获取相机的数据
#     camera_data = camera.data

#     # # 获取相机的焦距和感光元件尺寸
#     focal_length = camera_data.lens  # 焦距
#     sensor_width = camera_data.sensor_width  # 感光元件宽度
#     sensor_height = camera_data.sensor_height  # 感光元件高度
#     principal_point_x = camera_data.shift_x  # 主点在图像上的 x 坐标
#     principal_point_y = camera_data.shift_y  # 主点在图像上的 y 坐标

#     # # 计算相机的内参矩阵
#     print("focal_length",focal_length)
#     print("bpy.context.scene.render.resolution_x",bpy.context.scene.render.resolution_x)
#     print("bpy.context.scene.render.resolution_y",bpy.context.scene.render.resolution_y)
#     print("sensor_width",sensor_width)
#     print("sensor_height",sensor_height)
#     print("principal_point_x",principal_point_x)
#     print("principal_point_y",principal_point_y)
#     fx = focal_length * bpy.context.scene.render.resolution_x / sensor_width
#     fy = focal_length * bpy.context.scene.render.resolution_y / sensor_height
#     cx = bpy.context.scene.render.resolution_x * (0.5 - principal_point_x)
#     cy = bpy.context.scene.render.resolution_y * (0.5 - principal_point_y)

    
#     cam_intrinsics = np.array([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1],
#     ], dtype=np.float32)

#     print("cam_intrinsics",cam_intrinsics)

#     camera_location = camera.location
#     camera_rotation = camera.rotation_euler

#     print("camera_location",camera_location)
#     print("camera_rotation",camera_rotation)
#     # 构建相机的外参矩阵
#     rotation_matrix = camera_rotation.to_matrix().to_4x4()
#     translation_vector = Matrix.Translation(camera_location).to_4x4()
#     extrinsic_matrix = translation_vector @ rotation_matrix
#     extrinsic_matrix = np.array(extrinsic_matrix)[:3, :]

#     print("extrinsic_matrix1",extrinsic_matrix)

#     PP = cam_intrinsics @ extrinsic_matrix

#     print("PP",PP)
  

#     projection_matrix = camera.calc_matrix_camera(
#         bpy.context.evaluated_depsgraph_get(),
#         x=bpy.context.scene.render.resolution_x,
#         y=bpy.context.scene.render.resolution_y
#     )

#     print("projection_matrix2",projection_matrix)

def render_current_frame(path):
    import bpy

    # 设置渲染路径
    bpy.context.scene.render.filepath = path

    # 使用 Cycles 或 Eevee 渲染引擎
    bpy.context.scene.render.engine = 'CYCLES'  # 或 'BLENDER_EEVEE'

    # 设置背景为白色
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

    # 激活 Freestyle
    bpy.context.scene.render.use_freestyle = True

    # 获取当前视图层的 Freestyle 设置
    freestyle = bpy.context.scene.view_layers["View Layer"].freestyle_settings

    # 确保有一个线条样式集
    if not freestyle.linesets:
        lineset = freestyle.linesets.new(name="SketchLineSet")
    else:
        lineset = freestyle.linesets[0]

    lineset.qi_start = 1
    lineset.qi_end = 1

    # 获取或创建一个线条样式
    linestyle_name = "SketchLineStyle"
    if linestyle_name not in bpy.data.linestyles:
        linestyle = bpy.data.linestyles.new(name=linestyle_name)
    else:
        linestyle = bpy.data.linestyles[linestyle_name]

    # 设置铅笔画风格
    linestyle.color = (0.1, 0.1, 0.1)  # 浅灰色，接近铅笔颜色
    linestyle.thickness = 2.0  # 设置线条粗细
    linestyle.use_dashed_line = False  # 使用虚线效果
    # linestyle.dash1 = 5.0  # 虚线的线段长度
    # linestyle.gap1 = 3.0  # 虚线的间隔长度

    # # 添加噪声修饰器
    # if not any(mod.type == 'PERLIN_NOISE_1D' for mod in linestyle.geometry_modifiers):
    #     noise_modifier = linestyle.geometry_modifiers.new(name="PerlinNoise", type='PERLIN_NOISE_1D')
    #     noise_modifier.frequency = 10.0  # 调整频率
    #     noise_modifier.amplitude = 0.2  # 噪声振幅
    #     noise_modifier.octaves = 2  # 八度

    # # 添加粗糙度修饰器
    # if not any(mod.type == 'SPATIAL_NOISE' for mod in linestyle.geometry_modifiers):
    #     spatial_noise_modifier = linestyle.geometry_modifiers.new(name="SpatialNoise", type='SPATIAL_NOISE')
    #     spatial_noise_modifier.amplitude = 0.3  # 空间噪声振幅
    #     spatial_noise_modifier.scale = 15.0  # 噪声尺度

    # 应用线条样式到线条样式集
    lineset.linestyle = linestyle

    # 执行渲染，使用当前视口设置，并写入文件
    bpy.ops.render.render(use_viewport=True, write_still=True)

# def render_current_frame(path):
#     # 设置渲染路径
#     bpy.context.scene.render.filepath = path
    
#     # 使用Cycles或Eevee渲染引擎
#     bpy.context.scene.render.engine = 'CYCLES'  # 或 'BLENDER_EEVEE'

#     # 设置背景为白色
#     bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)

#     # 激活Freestyle
#     bpy.context.scene.render.use_freestyle = True

#     # # 配置Freestyle线条样式
#     # freestyle_settings = bpy.context.scene.view_layers["View Layer"].freestyle_settings
#     # lineset = freestyle_settings.linesets[0] if freestyle_settings.linesets else freestyle_settings.linesets.new(name="Freestyle LineSet")
    
#     # # 线条样式设置
#     # linestyle = lineset.linestyle
#     # linestyle.color = (0, 0, 0)  # 设置线条颜色为黑色
#     # linestyle.thickness = 1.5  # 设置线条粗细

#     # 获取当前视图层的Freestyle设置
#     freestyle = bpy.context.scene.view_layers["View Layer"].freestyle_settings
    
#     # 确保有一个线条样式集
#     if not freestyle.linesets:
#         lineset = freestyle.linesets.new(name="SketchLineSet")
#     else:
#         lineset = freestyle.linesets[0]
    
#     lineset.qi_start = 1
#     lineset.qi_end = 1

#     # 获取或创建一个线条样式
#     linestyle_name = "SketchLineStyle"
#     if linestyle_name not in bpy.data.linestyles:
#         linestyle = bpy.data.linestyles.new(name=linestyle_name)
#     else:
#         linestyle = bpy.data.linestyles[linestyle_name]
    
#     linestyle.color =  (0.0, 0.0, 0.0)  # 设置线条颜色为黑色
#     # 将线条样式指定给线条样式集
#     lineset.linestyle = linestyle

    
#     # 可以添加更多Freestyle的设置，比如线条的变化、样式等以模拟素描效果

#     # 执行渲染，使用当前视口设置，并写入文件
#     bpy.ops.render.render(use_viewport=True, write_still=True)

def render(npydata, frames_folder, *, mode, faces_path, gt=False,
           exact_frame=None, num=5, downsample=True,
           canonicalize=True, always_on_floor=False, denoising=True,
           oldrender=True,jointstype="mmm", res="high", init=True,
           joint_3d_traj=None,joint_ids=None,
           accelerator='gpu',device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res, denoising=denoising, oldrender=oldrender,accelerator=accelerator,device=device)

    is_mesh = mesh_detect(npydata)

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
            if joint_3d_traj is not None:
                joint_3d_traj = joint_3d_traj[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.2
        # npydata = prune_begin_end(npydata, perc)

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata, gt=gt, mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor)
    else:
        from .joints import Joints
        data = Joints(npydata, gt=gt, mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype,
                      joint_ids=joint_ids,
                      joint_3d_traj=joint_3d_traj
                      )

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    # show_traj(data.trajectory)

    # if len(joint_ids) > 1:
    #     show_traj(data.trajectory_2)

    # Create a floor
    # plot_floor(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render-1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render-1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    # bpy.ops.wm.save_as_mainfile(filepath="/Users/mathis/TEMOS_github/male_line_test.blend")
    # exit()

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path



