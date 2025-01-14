import numpy as np


class TEST:
    def __init__(self):
        self.center = np.array([0, 0, 0])

    def euler_angles_from_rotation_matrix(self, matrix):
        """从旋转矩阵计算欧拉角"""
        z = np.arctan2(-matrix[0, 1], matrix[0, 0])
        z = np.degrees(z)
        x = np.arctan2(-matrix[1, 2], matrix[2, 2])
        a = -matrix[1, 2] / np.sin(x)
        x = np.degrees(x)
        y = np.arctan2(matrix[0, 2], a)
        y = np.degrees(y)

        return x, y, z

    def normalize_vector(self, vector):
        """归一化向量"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def rotation_matrix_from_vectors(self, v1, v2, v3):
        """从三个向量构建旋转矩阵"""
        rotation_matrix = np.column_stack((v1, v2, v3))
        return rotation_matrix

    def rotate_coordinate(self, x, y, z, angle_x, angle_y, angle_z, reverse_turn=False):
        """旋转坐标点"""
        # 将角度从度转换为弧度
        angle_x = np.radians(angle_x)
        angle_y = np.radians(angle_y)
        angle_z = np.radians(angle_z)

        # 绕X、Y、Z轴的旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z
        R = R.transpose()

        # 原始点向量
        original_vector = np.array([x, y, z])

        # 将点平移到原点
        translated_point = original_vector - self.center

        # 旋转点
        rotated_point = R @ translated_point

        # 将点平移回中心
        new_point = rotated_point + self.center

        return new_point

    def set_coordinate_system(self):
        """设置坐标系"""
        key_points = [
            ('a', (1, 2, 3), (0, 0, 0), (0, 0, 0)),
            ('b', (2, 3, 4), (0, 0, 0), (0, 0, 0)),
            ('c', (4, 5, 6), (0, 0, 0), (0, 0, 0)),
            ('d', (4, 5, 7), (0, 0, 0), (0, 0, 0)),
        ]

        # 计算旋转后的点
        points = [self.rotate_coordinate(x, y, z, angle_x, angle_y, angle_z, True)
                  for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points]

        # 计算向量AB和CD
        vector_AB = points[1] - points[0]
        vector_CD = points[3] - points[2]

        # 计算水平面、冠状面和矢状面的法向量
        vector_axial = self.normalize_vector(np.cross(vector_AB, vector_CD))  # 水平面的法向量
        vector_coronal = self.normalize_vector(np.cross(vector_CD, vector_axial))  # 冠状面的法向量
        vector_sagittal = self.normalize_vector(np.cross(vector_coronal, vector_axial))  # 矢状面的法向量

        # 构建旋转矩阵
        rotation_matrix = self.rotation_matrix_from_vectors(vector_sagittal, vector_coronal, vector_axial)
        print(f"rotation_matrix\n{rotation_matrix}")

        # 计算欧拉角
        euler_angles = self.euler_angles_from_rotation_matrix(rotation_matrix)
        print(f"euler_angles\n{euler_angles}")

        # 使用欧拉角重新旋转点
        rotated_points_with_euler = [
            self.rotate_coordinate(x, y, z, euler_angles[0], euler_angles[1], euler_angles[2], True)
            for (name, (x, y, z), (angle_x, angle_y, angle_z), (phy_x, phy_y, phy_z)) in key_points
        ]

        return rotated_points_with_euler


def main():
    """主函数"""
    # 创建 TEST 类的实例
    test_instance = TEST()

    # 调用 set_coordinate_system 方法
    rotated_points = test_instance.set_coordinate_system()

    # 打印旋转后的点
    print("Rotated Points with Euler Angles:")
    for point in rotated_points:
        print(point)


if __name__ == "__main__":
    main()