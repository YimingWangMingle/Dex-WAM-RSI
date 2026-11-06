# vendored 前端依赖

本目录存放监控前端使用的第三方库（离线 vendor，局域网部署不依赖 CDN）。

## echarts.min.js

- 项目：Apache ECharts
- 版本：5.6.0（官方 `echarts@5.6.0` npm 包内 `dist/echarts.min.js`，UMD 全量构建）
- 许可：Apache-2.0（全文见 `echarts.LICENSE.txt`）
- SHA256：`bf4a223524e40b77c304bec67e1222cf551f14880cf42c69dc046558e11c07b1`
- 用途：监控前端全部图表（折线/散点/堆叠柱/饼图/小倍数网格/custom series 时间轴），
  经 `<script src="/static/vendor/echarts.min.js">` 暴露全局 `window.echarts`。
- 升级方式：`npm pack echarts@<新版>`，解包取 `dist/echarts.min.js` 覆盖本文件，
  并更新本 README 的版本号与 SHA256。
