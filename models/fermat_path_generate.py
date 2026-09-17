import torch
import numpy as np

def generate_fermat_spiral_indices(H, W):
    # 定义参数
    c_x = (W - 1) / 2
    c_y = (H - 1) / 2
    golden_angle_rad = np.radians(137.508)
    N = H * W
    
    # 计算最大半径
    r_max = np.sqrt((W-1 - c_x)**2 + (H-1 - c_y)**2)
    a = r_max / np.sqrt(N - 1)
    
    # 生成螺旋点
    indices = np.arange(N)
    theta = indices * golden_angle_rad
    r = a * np.sqrt(indices)
    x_spiral = r * np.cos(theta)
    y_spiral = r * np.sin(theta)
    
    # 生成网格点
    grid_x = []
    grid_y = []
    for i in range(H):
        for j in range(W):
            grid_x.append(i - c_x)
            grid_y.append(j - c_y)
    grid_x = np.array(grid_x)
    grid_y = np.array(grid_y)
    
    # 初始化结果索引和已使用标记
    result_indices = []
    used_grid_points = set()
    
    # 逐个处理螺旋点，为每个螺旋点找到最近的未使用网格点
    for k in range(N):
        # 计算所有网格点到当前螺旋点的距离
        distances = (grid_x - x_spiral[k])**2 + (grid_y - y_spiral[k])**2
        
        # 按距离排序的网格点索引
        sorted_grid_indices = np.argsort(distances)
        
        # 找到最近的未使用网格点
        for grid_idx in sorted_grid_indices:
            if grid_idx not in used_grid_points:
                result_indices.append(grid_idx)
                used_grid_points.add(grid_idx)
                break
                
        # 如果已经找到所有网格点，就可以提前结束
        if len(result_indices) == N:
            break
    
    # 检查是否所有网格点都被包含
    if len(result_indices) < N:
        # 添加任何漏掉的网格点
        for i in range(N):
            if i not in used_grid_points:
                result_indices.append(i)
                used_grid_points.add(i)
    
    # 检查结果的唯一性和完整性
    assert len(result_indices) == N, f"索引数量不正确: {len(result_indices)} vs 期望 {N}"
    assert len(set(result_indices)) == N, "索引中存在重复"
    
    return torch.tensor(result_indices, dtype=torch.long)

# 用法示例
H, W = 64, 64
spiral_indices = generate_fermat_spiral_indices(H, W)
column_vector = np.arange(64*64)

# 验证输出的唯一性和完整性
unique_count = len(torch.unique(spiral_indices))
print(f"生成的索引数量: {len(spiral_indices)}")
print(f"唯一索引数量: {unique_count}")
print(f"是否所有索引都唯一: {unique_count == H*W}")

matrix = np.zeros((64*64, 64*64), dtype=int)
matrix[column_vector, spiral_indices] = 1

np.save('fermat_spiral_eye.npy', matrix)
np.save('fermat_despiral_eye.npy', np.transpose(matrix))

spiral_indices_r = spiral_indices.flip(0)
matrix = np.zeros((64*64, 64*64), dtype=int)
matrix[column_vector, spiral_indices_r] = 1

np.save('fermat_despiral_r_eye.npy', np.transpose(matrix))
