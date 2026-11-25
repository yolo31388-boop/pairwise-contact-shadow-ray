# pairwise-contact-shadow-ray

## 领域

游戏、图形与多媒体

## 主题

接触阴影射线

## 说明

当前帧深度、屏幕尺度步长与最暗合并。

初始状态下 `graphics/contact_shadow_ray.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
