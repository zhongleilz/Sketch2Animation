from textwrap import wrap
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib

import mld.data.humanml.utils.paramUtil as paramUtil

skeleton = paramUtil.t2m_kinematic_chain


# def plot_3d_motion(save_path: str, joints: np.ndarray, title: str,
#                    figsize: tuple[int, int] = (3, 3),
#                    fps: int = 120, radius: int = 3, kinematic_tree: list = skeleton,
#                    hint: Optional[np.ndarray] = None) -> None:

#     title = '\n'.join(wrap(title, 20))

#     def init():
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
#         fig.suptitle(title, fontsize=10)
#         ax.grid(b=False)

#     def plot_xzPlane(minx, maxx, miny, minz, maxz):
#         # Plot a plane XZ
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     data *= 1.3  # scale for visualization
#     if hint is not None:
#         mask = hint.sum(-1) != 0
#         hint = hint[mask]
#         hint *= 1.3

#     fig = plt.figure(figsize=figsize)
#     plt.tight_layout()
#     ax = p3.Axes3D(fig)
#     init()
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)
#     colors = ["#DD5A37", "#D69E00", "#B75A39", "#DD5A37", "#D69E00",
#               "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00", "#FF6D00",
#               "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", "#DDB50E", ]

#     frame_number = data.shape[0]

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     if hint is not None:
#         hint[..., 1] -= height_offset
#     trajec = data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     def update(index):
#         # ax.lines = []
#         # ax.collections = []
#         ax.clear()
#         ax.view_init(elev=120, azim=-90)
#         ax.dist = 7.5
#         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
#                      MAXS[2] - trajec[index, 1])

#         if hint is not None:
#             ax.scatter(hint[..., 0] - trajec[index, 0], hint[..., 1], hint[..., 2] - trajec[index, 1], color="#80B79A")

#         for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
#                       color=color)

#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
#     ani.save(save_path, fps=fps)
#     plt.close()

def plot_2d_motion(save_path, joints, title, figsize=(3, 3), fps=120, radius = 3,kinematic_tree=skeleton, hint=None,
                   vis_mode='default', gt_frames_2d=None,joint_id=0):
    matplotlib.use('Agg')  # 使用无显示的后端

    title = '\n'.join(title.split())  # 处理标题

    fig, ax = plt.subplots(figsize=figsize)
    plt.tight_layout()
    colors = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"] 
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    if hint is not None:
        mask = hint.sum(-1) != 0
        hint = hint[mask]
        hint = hint * 1.3
    
    joints = joints * 1.3
    trajec2 = joints[:, joint_id, [0, 1]]

    if gt_frames_2d is not None:
        gt_frames_2d *= 1.3

    def init():
        # 设置2D视图的显示范围
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        fig.suptitle(title, fontsize=10)
        ax.grid(False)  # 不显示网格

    def update(index):
        ax.clear()  # 清除之前的帧
        init()  # 重新初始化视图

        if hint is not None:
            ax.scatter(hint[..., 0], hint[..., 1],s= 5, color="#80B79A")
        

        if index > 1:
            # if joint_id != 0:
            ax.scatter(trajec2[:index, 0], trajec2[:index, 1] , s= 5,
                    color='blue')

        used_colors = colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot(joints[index, chain, 0], joints[index, chain, 1], linewidth=linewidth, color=color)
        
        
        if gt_frames_2d is not None:
            used_colors = colors_orange
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                
                if index == 0:
                    ax.plot(gt_frames_2d[index, chain, 0], gt_frames_2d[index, chain, 1], linewidth=linewidth,
                            color=color)



        plt.axis('off')  # 不显示坐标轴

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(joints), init_func=init, repeat=False)
    
    # 保存动画为视频
    ani.save(save_path, writer='ffmpeg', fps=fps)

    plt.close(fig)



def plot_3d_motion(save_path,joints,title,figsize=(3, 3),fps=120, radius=3,kinematic_tree=skeleton,hint=None,
                   vis_mode='default', gt_frames=None,joint_id=0):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    data *= 1.3
    
    if hint is not None:
        mask = hint.sum(-1) != 0
        hint = hint[mask]
        hint *= 1.3

    if gt_frames is not None:
        gt_frames *= 1.3
    fig = plt.figure(figsize=figsize)
    
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset

    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    trajec2 = data[:, joint_id, [0, 2, 1]]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        # ax.lines = []
        # ax.collections = []
        ax.clear()

        # ax.view_init(elev=120, azim=-90)
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        
        if hint is not None:
            ax.scatter(hint[..., 0] - trajec[index, 0], hint[..., 1], hint[..., 2] - trajec[index, 1], color="#80B79A")
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        if index > 1:
            # if joint_id == 0:
            #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
            #             trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
            #             color='blue')
            # else:
            ax.plot3D(trajec2[:index, 0], trajec2[:index, 2],
                    trajec2[:index, 1] , linewidth=1.0,
                    color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        used_colors = colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        
        if gt_frames is not None:
            used_colors = colors_blue
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                
                if index == 0:
                    ax.plot3D(gt_frames[index, chain, 0], gt_frames[index, chain, 1], gt_frames[index, chain, 2], linewidth=linewidth,
                            color=color)
        
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()