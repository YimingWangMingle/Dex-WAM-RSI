# 可选历史兼容层

正式框架不导入此目录。不交付 `compat/` 时，可以使用新目录、新模块名和正常 CLI；
旧模块名、历史脚本导入及原实验对照工具不属于无兼容层发行版的支持范围。

## 显式启用

在仓库根运行旧脚本：

```bash
python -m compat /path/to/legacy_script.py
```

或在确实需要旧导入的程序入口中调用：

```python
from compat import enable
enable()
from runtime.robodojo_a2v import A2VReviewMixin
```

普通 `python -m rsi episode/campaign/platforms` 不需要启用兼容。
`python -m compat -m <模块名>` 可运行使用旧导入的普通脚本模块；别名模块本身不是新的 CLI。

## 内容

- `module_aliases.json`、`imports.py`：旧模块名到实际模块的显式映射，新旧导入共享模块身份。
- `legacy/adapters.py`、`legacy/runtime/`：仅在 enable 后加入搜索路径的历史包入口。
- `vendor.py`：保留旧 vendor 加载函数；正常模型推理不再调用。
- `experiment_protocol_worker.py`：原实验与新代码的独立进程对照，保留参考端旧导入。
- `tests/`：兼容层专属测试；与正式框架测试分开。

RoboTwin 在线模块已经删除，不提供兼容。旧源码的物理文件路径不保证存在。
根目录模型软链接已经删除。models、utils、wan 等旧导入仅在 enable 后映射到
thinker_wm 下的同一模块；旧物理配置路径不恢复，需使用 third_party/Thinker-WM/ 下的实际路径。
launchers/ 保存历史原生实验、定批评测和监控启动脚本；这些脚本可能指向冻结机器路径，
需要按原实验环境使用，不承诺新机器直接运行。新任务使用 python -m rsi。

## 无兼容层验证

```bash
python scripts/test_without_compat.py --output /absolute/path/outside-repo/optional-compat.json
python -m unittest discover -s compat/tests -v
```

第一条导出一份不含 compat 的临时副本，排除原仓库搜索路径，并拦截 compat/runtime/adapters
旧模块导入，再执行框架、传输和模型缓存 CPU 回归。第二条单独验证历史导入身份。
这些检查不调用真实模型服务或仿真器，不替代 GPU 数值与完整任务成功率验证。
