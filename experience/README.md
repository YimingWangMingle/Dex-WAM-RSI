# 经验包与 HMR / ICL

`packages.py` 管理封装、哈希校验、catalog、按任务选择及单局冻结工作区。
Markdown 是供 Agent 读取的经验内容，不是可执行技能或当前实测证据。

- `templates/`：可复制填写的 HMR / ICL 模板。
- `examples/hmr/`、`examples/icl/`：未封装的教学正文。
- `examples/packages/`：上述正文的封存示例，包含 manifest 与 payload。
- `examples/catalog.json`：可直接加载的示例索引，使用相对路径和真实文件哈希。

在仓库根验证示例：

```python
from experience.packages import ExperienceCatalog
catalog = ExperienceCatalog.load("experience/examples/catalog.json")
selection = catalog.select("fixture_move_hmr")
package = catalog.package(selection.bundle_id)
print(package.identity())
```

ICL 对应任务名为 `fixture_move_icl`。两个示例都是虚构教学内容，不能用于报告真实实验结果。
实际接线仍需 `experience_catalog`、`experience_task` 和按技能配置的 `skill_experience`；
单独有 Markdown 文件不会自动注入模型。详见 [使用指南](../docs/EXPERIENCE_GUIDE.md)。

真实实验经验放在独立输出目录或经验库。使用 `build_package` 封装到新目录，再用
`write_catalog` 发布新版本；不要直接修改已有 payload 或手工绕过哈希校验。
每局读取冻结副本。运行中新产生的经验单独保存为候选，外层 RSI 评估后才进入下一版本。
