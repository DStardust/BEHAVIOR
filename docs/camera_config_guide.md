# 全局摄像头配置参考文档

> **唯一允许的摄像头放置方法：墙角法 + 墙面居中法。** h_offset=0。
> - **大房间**（对角线 > 3m）：墙角法（SW/SE/NW/NE），沿对角线向内偏移
> - **小房间**（对角线 ≤ 3m）：墙面居中法，摄像头在短边墙中心，v_angle 更大（更俯视）

---

## 1. 核心方法

### 1.1 房间角点获取

**关键发现**：seg_map 的房间标签（`room_ins_name_to_ins_id`）与 3D 场景的房间名不匹配。必须通过 3D 物体位置反查 seg_map 像素坐标来获取正确的墙面位置。

```python
seg_map = env.scene.seg_map
room_positions = defaultdict(list)
for obj in get_all_scene_objects(env.scene):
    pos, _ = obj.get_position_orientation()
    rooms = getattr(obj, 'in_rooms', None)
    if not rooms: continue
    if isinstance(rooms, str): rooms = [rooms]
    for r in rooms: room_positions[r].append(np.array(pos))

def get_room_corners(positions):
    """通过 3D 物体位置 → seg_map 像素坐标 → 世界坐标获取房间墙角"""
    pixels = []
    for pos in positions:
        px = seg_map.world_to_map(torch.tensor([pos[0], pos[1]], dtype=torch.float32))
        pixels.append(px.numpy())
    pixels = np.stack(pixels)
    px_min = pixels.min(axis=0).astype(int)
    px_max = pixels.max(axis=0).astype(int)
    map_h, map_w = seg_map.room_ins_map.shape
    px_min = np.clip(px_min, 0, [map_w-1, map_h-1])
    px_max = np.clip(px_max, 0, [map_w-1, map_h-1])
    sw = seg_map.map_to_world(torch.tensor([px_min[0], px_min[1]], dtype=torch.float32)).cpu().numpy()
    ne = seg_map.map_to_world(torch.tensor([px_max[0], px_max[1]], dtype=torch.float32)).cpu().numpy()
    return {
        'SW': sw,
        'NE': ne,
        'NW': np.array([sw[0], ne[1]]),
        'SE': np.array([ne[0], sw[1]]),
    }
```

### 1.2 摄像头位置

#### 墙角法（大房间，对角线 > 3m）

选择某个墙角，沿对角线向内偏移，高度 2.4m：

```python
INWARD = 0.3   # 向内偏移量（米），小房间自动减小为 diag_len * 0.1
HEIGHT = 2.4   # 摄像头高度（米）

corner = corners["SW"]      # 选择的墙角
opposite = corners["NE"]    # 对角点
diagonal = opposite - corner
diag_len = np.sqrt(diagonal[0]**2 + diagonal[1]**2)
inward = min(INWARD, diag_len * 0.1)

cam_pos = np.array([
    corner[0] + (diagonal[0] / diag_len) * inward,
    corner[1] + (diagonal[1] / diag_len) * inward,
    HEIGHT,
])
```

#### 墙面居中法（小房间，对角线 ≤ 3m）

摄像头在短边墙中心，高度 2.2m，向内偏移 0.2m，v_angle=45°（更俯视）：

```python
# 4 面墙：SW-SE, SE-NE, NE-NW, NW-SW
# 选最短的墙
walls = [
    ("SW", "SE", corners["SW"], corners["SE"]),
    ("SE", "NE", corners["SE"], corners["NE"]),
    ("NE", "NW", corners["NE"], corners["NW"]),
    ("NW", "SW", corners["NW"], corners["SW"]),
]
shortest = min(walls, key=lambda w: np.linalg.norm(w[3][:2] - w[2][:2]))

c1, c2 = shortest[2], shortest[3]
wall_center = (c1 + c2) / 2
wall_dir = c2 - c1
wall_dir = wall_dir / np.linalg.norm(wall_dir)
normal = np.array([-wall_dir[1], wall_dir[0]])  # 向内法线

cam_pos = np.array([
    wall_center[0] + normal[0] * 0.2,
    wall_center[1] + normal[1] * 0.2,
    2.2,  # 小房间摄像头稍低
])
```

### 1.3 摄像头朝向

**Isaac Sim 约定**：
- 默认朝向 [0,0,0,1]：摄像头朝下 (-Z)
- pitch=90° 绕 X：摄像头水平（朝 +Y）
- yaw 绕 Z：逆时针旋转

**四元数构建**：先 pitch 绕 X，再 yaw 绕 Z。h_offset=0。

```python
def quat_multiply(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2,
        w1*w2-x1*x2-y1*y2-z1*z2,
    ], dtype=np.float32)

# 墙角法：沿对角线方向
diag_angle = math.degrees(math.atan2(diagonal[1], diagonal[0]))
base_yaw = diag_angle - 90.0
yaw = math.radians(base_yaw)  # h_offset=0

# 墙面居中法：沿法线方向
normal_angle = math.degrees(math.atan2(normal[1], normal[0]))
base_yaw = normal_angle - 90.0
yaw = math.radians(base_yaw)  # h_offset=0

# 通用 pitch
pitch = math.radians(90.0 - v_angle)

cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
q_pitch = np.array([sp, 0, 0, cp], dtype=np.float32)  # 绕 X 轴
q_yaw   = np.array([0, 0, sy, cy], dtype=np.float32)  # 绕 Z 轴
orientation = quat_multiply(q_yaw, q_pitch)           # 先 pitch 后 yaw
```

---

## 2. 参数选择规则

| 房间大小 | 方法 | 默认 v_angle | 高度 | inward |
|---------|------|-------------|------|--------|
| 大房间（对角线 > 3m） | 墙角法 | 30° | 2.4m | 0.3m |
| 小房间（对角线 ≤ 3m） | 墙面居中法 | 45° | 2.2m | 0.2m |

> **v_angle 调整**：增大 → 更俯视（看地板），减小 → 更水平（看墙壁）。小房间需要更大 v_angle 以看到更多地面。

### 2.1 Env-A/B/C 多视角覆盖契约

任务实例生成统一使用以下规则：

1. 每个合格样本必须保存 2–3 个位置不同的全局摄像头，默认参数为
   `--min-global-cameras 2 --max-global-cameras 3`。
2. 所有 solution plan 涉及的目标物品必须至少被一个全局摄像头看到；机器人也必须至少被一个全局摄像头看到。机器人自带相机的可见性不能替代这两个全局视角门槛。
3. 约 50% 的样本请求 3 相机重复覆盖。第二台相机优先完成机器人或目标物品的重复覆盖；重复覆盖一旦达成，第三台优先放在尚未使用的非目标房间。`redundant_coverage_achieved` 记录是否实际形成重复覆盖。
4. 补充摄像头可以放在目标房间或非目标房间，也不要求同时看到机器人或任务物品；它必须原样使用本指南的官方墙角/墙面固定朝向，禁止再次做动态 look-at。朝墙、朝天或嵌入结构的画面不属于有效补充视角。
5. 同一房间允许布置多个摄像头，但必须使用不同的官方墙角/墙面位置；候选按位置去重，只改变朝向而位置相同不算两个摄像头。

生成结果在 `validation.camera_coverage` 中记录相机数量、目标逐物体覆盖次数、机器人覆盖次数、是否请求及达成重复覆盖。`audit_deltasg_outputs.py` 会拒绝少于 2 个或多于 3 个全局相机、目标未被全局相机覆盖、或没有全局相机看到机器人的样本。

---

## 3. 当前场景配置

### 3.1 Rs_int

| 房间 | 方法 | 墙角/墙面 | v_angle | 说明 |
|------|------|----------|---------|------|
| living_room_0 | 墙角 | SW | 30° | 客厅全景 ✓ |
| bedroom_0 | 墙角 | SE | 45° | 卧室全景 ✓ |
| bathroom_0 | 墙角 | NE | 45° | 避开淋浴间 ✓ |
| kitchen_0 | — | — | — | 放弃：顶柜遮挡 |
| entryway_0 | 墙面 | 最短边 | 45° | 小房间 |

### 3.2 Benevolence_0_int

| 房间 | 方法 | 墙角/墙面 | v_angle | 说明 |
|------|------|----------|---------|------|
| bathroom_0 | 墙面 | 最短边 | 45° | 小房间 |
| empty_room_0 | 墙角 | SW | 30° | 待确认 |
| corridor_0 | 墙面 | 最短边 | 45° | 窄走廊 |
| entryway_0 | — | — | — | 物体太少，无法放置 |

---

## 4. 使用方法

1. 运行 `python code/capture_all_rooms.py --scene <场景名>` 生成所有房间的 8 视角预览
2. 查看预览图，选择每个房间的最佳视角（墙角/墙面 + v_angle）
3. 更新本文档的配置表
4. 后续 `_compute_global_camera_pose` 和 `capture_batch10_camera.py` 自动读取配置

---

## 5. 专家求解阶段的传感器生命周期

任务生成阶段只使用上述官方候选位姿做视锥投影和 PhysX 射线可见性检查，避免在高频生成循环中反复创建 Replicator 图。

专家求解阶段必须保存真实渲染结果，并遵守以下约束：

1. 机器人主相机使用机器人自带的官方 `VisionSensor`。
2. 机器人主相机是进程内唯一的实例分割 annotator；语义分割由该官方实例分割标签稳定派生，用于精细操作 bbox 验证。
3. 每个已选全局位姿分别创建一个固定 RGB `VisionSensor`。全局可见性和 bbox 复用任务生成阶段已经通过视锥投影与 PhysX 射线检查的几何证据，并按实际输出分辨率缩放。
4. 每个采样事件保存机器人 RGB、实例分割、语义分割、bbox，以及所有全局相机 RGB 和几何可见性证据；审计必须实际解码这些文件并拒绝空白、损坏或全背景结果。

原因：Isaac Sim 5.1 在移动已挂载实例分割 annotator 的相机后，或在同一进程挂载第二个实例分割流后，可能在 `SyntheticData._post_process_graph_tick` 中发生进程级崩溃。唯一分割流加固定 RGB 全局相机既保留官方位姿策略，也避免该生命周期问题。
