# VSCode Debug 配置说明

## 文件说明

- `launch.json.example`: 调试配置模板（提交到 Git）
- `launch.json`: 实际使用的配置（已被 `.gitignore` 忽略）

## 首次设置

如果您是第一次克隆项目，需要复制模板文件：

```bash
# Windows PowerShell
Copy-Item .vscode/launch.json.example .vscode/launch.json

# Linux/Mac
cp .vscode/launch.json.example .vscode/launch.json
```

然后根据您的实际环境修改 `launch.json` 中的路径。

## 为什么这样做？

- ✅ **避免冲突**: 每个人的 `launch.json` 可能有不同的路径配置
- ✅ **保留模板**: `launch.json.example` 提供了标准配置参考
- ✅ **灵活修改**: 可以根据本地环境自由修改 `launch.json`
- ✅ **不影响他人**: 您的修改不会影响其他协作者

## 更新模板

如果您添加了新的有用的调试配置，想要分享给团队：

1. 更新 `launch.json.example`
2. 提交到 Git
3. 通知团队成员更新他们的 `launch.json`

## 常见问题

**Q: 我修改了 launch.json，会被提交吗？**  
A: 不会，它已经在 `.gitignore` 中被忽略了。

**Q: 如何获取最新的配置模板？**  
A: 从 Git 拉取最新的 `launch.json.example`，然后手动合并到您的 `launch.json`。

**Q: 我可以直接使用 launch.json.example 吗？**  
A: 可以，但建议复制一份为 `launch.json`，这样更新模板时不会影响您的配置。
