# 接入新数据集：Episode 数据源插件契约

本文说明如何在不修改 motus_data 库代码的前提下接入一个新的数据集源，
以及项目维护者如何把第三方实现收编为内置数据源。适用范围是 Episode
Catalog 链路的三个入口：

- `open_dataset()`：通用 Episode 访问；
- `open_training_data()`：单源训练数据；
- `open_mixed_training_data()`：多源加权混训。

旧的 config 驱动 `build_dataset()` 体系（`motus_data/factory.py`）不在
本文范围内。

可直接运行的最小完整模板见 `examples/custom_episode_source.py`。

## 第三方接入：四步

### 1. 实现 `EpisodeSourceReader` 子类（统一接口）

```python
from motus_data import EpisodeSourceReader

class MySourceReader(EpisodeSourceReader):
    source_dataset = "my_source"   # 必须与 Catalog metadata 的 source 一致
```

必须实现的方法：

| 方法 | 返回 |
| --- | --- |
| `schemas(record)` | `Mapping[str, ModalitySchema]`，四个模态 video/state/action/text |
| `read_video(record, indices, *, output_format, video_size)` | 帧张量 |
| `read_state(record, indices)` | `(len(indices), state_dim)` float32 |
| `read_action(record, indices)` | `(len(indices), action_dim)` float32 |

可选覆写：`read_state_action()`（state/action 同文件时避免读两次，默认
实现是分调 `read_state`/`read_action`）、`iter_video_chunks()`。

reader 拿到的是**归一化后的 Catalog record**：

- 每个模态为 `{"count"/"frame_count": int, "locator": {...}, "metadata": {...}}`；
- `locator` 含 `storage_backend`（`"file"`/`"hdf5"`/`"tar_member"`）、
  `uri`，tar 另有 `member`/`offset_data`/`size`；
- 文本在 `record["text"]["items"]`；
- `record["source_dataset"]` 取自 Catalog metadata 的 `source`。

参考内置实现：`motus_data/source_readers.py`（file/HDF5 三族）和
`motus_data/repository.py` 的 `AgibotBetaEpisodeReader`（tar）。

### 2. 构建契约合规的 Catalog

三个文件（以 `my_source.episode_catalog.jsonl` 为例）：

- `my_source.episode_catalog.jsonl`：每行一个 `motus_episode_catalog_v2`
  record（`episode_id`/`robot`/`video`/`state`/`action`/`texts`，
  `episode_id` 必须以 `"my_source:"` 开头）；
- `my_source.episode_catalog.meta.json`：metadata，`source: "my_source"`
  必须与第 1 步注册名一致，`sha256` 为 Catalog 文件的实际 SHA256，
  `episodes` 为实际行数；
- `my_source.episode_catalog.index.sqlite`：调用
  `motus_data.build_catalog_offset_index(catalog_path)` 生成，构建时会
  校验契约、行数和 SHA256。

契约细节见 `docs/EPISODE_REPOSITORY_V1.md` 与
`motus_data/index_contracts.py`。`tools/build_episode_catalog.py` 的
`--family` 仅覆盖内置三族（agibot/robotwin/robotmind）；新数据源的离线
Catalog 构建脚本由接入方自备，只需产出上述契约合规文件。

### 3. 注册 reader（import 时执行一次）

```python
from motus_data import register_episode_source_reader

register_episode_source_reader("my_source", MySourceReader)
```

- 可传类或实例；传类时每个 `EpisodeRepository` 构造会零参实例化一份；
- 名称规范化（小写），重复注册抛 `KeyError`，`replace=True` 覆盖；
- 可用 `motus_data.episode_source_readers()` 查看当前已注册名。

### 4. 注册 release，然后按名字打开

```python
from motus_data import (
    EpisodeDatasetRelease,
    open_dataset,
    register_episode_dataset_release,
)

register_episode_dataset_release(
    EpisodeDatasetRelease(
        dataset="my_dataset",
        release="v1",
        catalog_path="/abs/path/my_source.episode_catalog.jsonl",
        index_path="/abs/path/my_source.episode_catalog.index.sqlite",
    ),
    default=True,
)
data = open_dataset("my_dataset")   # 不需要 readers=
```

`open_training_data("my_dataset", ...)` 与 `open_mixed_training_data(...)`
同样自动解析。reader 解析优先级：**调用方 `readers=` 映射 > 全局注册表**
（含内置 reader）；`readers=` 仅用于一次性覆盖，日常不需要。

## 维护者收编清单

第三方交付物就是一个文件（reader 类 + 注册行 + 构建说明）。收编为内置：

1. 把 reader 类和 `register_episode_source_reader(...)` 行原样复制进
   `motus_data/source_readers.py`（或新建模块，并在
   `motus_data/repository.py` 的 `_ensure_builtin_source_readers()` 中
   import 该模块）；
2. 在 `motus_data/releases.yaml` 追加该数据集的 release 条目；
3. `python -m pytest tests/ -q` 全绿后合入。

内置 reader 与第三方 reader 形态完全一致（一个类 + 一行注册），收编
不需要改动 reader 代码本身，也不需要改 `EpisodeRepository`。

## 排查

- `no Episode source reader registered for 'x'; registered=[...]`：
  Catalog metadata 的 `source` 与第 3 步注册名不一致（或第三方模块
  尚未 import，注册未执行）。
- `unknown release=... for dataset=...`：release 未注册或拼写不一致，
  用 `motus_data.episode_dataset_releases()` 查看。
