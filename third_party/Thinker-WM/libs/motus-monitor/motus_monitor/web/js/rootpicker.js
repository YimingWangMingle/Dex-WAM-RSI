/* Motus Monitor 前端 —— 查看路径选择弹窗（本机只读浏览 / 经 SSH 浏览远端 H-Server）。
 * 从旧 web/app.js:1590-1760 逐字迁出，沿用 index.html 的 #root-picker 静态 DOM
 * （index.html:42-71 保留不动），行为不变。
 */
import { api } from "./api.js";
import { util } from "./util.js";

/* ------------------------------------------------------------ 顶栏查看路径标签 */
export function refreshDataRootLabel() {
  return api.getOrNull("/api/data_root").then(function (info) {
    var btn = document.getElementById("data-root-btn");
    if (!info || !info.data_root) { btn.textContent = "…"; return; }
    btn.textContent = "📂 查看目录: " + info.data_root;
    var tips = "监控查看路径（点击切换）\n当前: " + info.data_root;
    if (info.startup_data_root && info.startup_data_root !== info.data_root) {
      tips += "\n启动目录: " + info.startup_data_root;
    }
    if ((info.sync_local_dirs || []).length) {
      tips += "\n同步写入: " + info.sync_local_dirs.join(", ");
    }
    btn.title = tips;
  });
}

export function initRootPicker() {
  var overlay = document.getElementById("root-picker");
  var listEl = document.getElementById("root-picker-list");
  var curEl = document.getElementById("root-picker-cur");
  var hintEl = document.getElementById("root-picker-hint");
  var descEl = document.getElementById("root-picker-desc");
  var chooseBtn = document.getElementById("root-picker-choose");
  var modeLocalBtn = document.getElementById("root-picker-mode-local");
  var modeRemoteBtn = document.getElementById("root-picker-mode-remote");
  var current = null;
  var mode = "local";           /* local | remote（经 SSH 浏览 H-Server 等远端） */
  var remoteName = null;        /* 参考 ssh 远端（复用其 target/ssh_args） */
  var sshRemotes = [];

  var DESC_LOCAL = "切换查看根 = 选择查看哪个监控数据目录下的 run 记录（只读）。只看你自己 workspace 的视图，不影响其他用户、各同步任务的写入目录与训练。";
  var DESC_REMOTE = "远端模式 = 经 SSH 浏览远端（H-Server）的目录；选中后把该目录加入你自己的 workspace 同步（与其他用户互不影响），查看根随之切换，数据拉取到本机镜像后 run 自动出现。";

  function render(info) {
    current = info.path;
    curEl.textContent = (mode === "remote" ? "[远端] " : "") + info.path;
    document.getElementById("root-picker-up").disabled = !info.parent;
    listEl.innerHTML = "";
    var dirs = info.dirs || [];
    if (!dirs.length) {
      listEl.appendChild(util.el("div", "root-picker-empty", { text: "（没有子目录）" }));
    }
    dirs.forEach(function (d) {
      var item = util.el("div", "root-picker-item");
      item.appendChild(util.el("span", "dir-icon", { text: "▸" }));
      item.appendChild(util.el("span", null, { text: d.name + "/" }));
      if (d.has_runs_dir) item.appendChild(util.el("span", "dir-tag", { text: "含 runs/" }));
      item.addEventListener("click", function () {
        browse(info.path.replace(/\/+$/, "") + "/" + d.name);
      });
      listEl.appendChild(item);
    });
    hintEl.textContent = info.looks_like_data_root
      ? (mode === "remote" ? "该远端目录含 runs/，可加入同步" : "该目录含 runs/，可作为监控数据根")
      : "";
  }

  function browse(path) {
    hintEl.textContent = mode === "remote" ? "远端加载中（SSH）…" : "加载中…";
    var url = "/api/browse?path=" + encodeURIComponent(path);
    if (mode === "remote") url += "&remote=" + encodeURIComponent(remoteName || "1");
    api.get(url)
      .then(render)
      .catch(function (err) { hintEl.textContent = err.message || "无法读取该目录"; });
  }

  /* 本机：切换查看根（整页重载最干净：各 tab 缓存与选中 run 都基于旧根） */
  function switchTo(path) {
    hintEl.textContent = "切换到 " + path + " …";
    api.post("/api/data_root", { path: path }).then(function () {
      window.location.reload();
    }).catch(function (err) {
      hintEl.textContent = err.message || "切换失败";
    });
  }

  /* 远端：把该目录加为自己 workspace 的同步远端（查看根随之切换） */
  function addRemote(path) {
    hintEl.textContent = "加入同步 " + path + " …";
    api.post("/api/remotes", { path: path, remote: remoteName }).then(function (resp) {
      hintEl.textContent = (resp && resp.note) || "已加入同步。";
      /* 服务端已把本用户查看根切到 workspace 数据目录：刷新顶栏标签 */
      if (resp && !resp.already) refreshDataRootLabel();
    }).catch(function (err) {
      hintEl.textContent = err.message || "加入同步失败";
    });
  }

  function knownItem(r, onClick) {
    var item = util.el("div", "root-picker-known-item");
    item.appendChild(util.el("span", "kr-path", { text: r.path }));
    var meta = (r.run_count || 0) + " runs";
    if (r.latest_ts) meta += " · 最近活跃 " + util.fmtAgo(r.latest_ts);
    item.appendChild(util.el("span", "kr-meta", { text: meta }));
    item.addEventListener("click", onClick);
    return item;
  }

  function loadKnownRoots() {
    var localWrap = document.getElementById("root-picker-known");
    var localList = document.getElementById("root-picker-known-list");
    var remoteWrap = document.getElementById("root-picker-remote-known");
    var remoteList = document.getElementById("root-picker-remote-known-list");
    api.getOrNull("/api/known_roots").then(function (info) {
      var roots = (info && info.roots) || [];
      localWrap.classList.toggle("hidden", !roots.length);
      localList.innerHTML = "";
      roots.forEach(function (r) {
        localList.appendChild(knownItem(r, function () { switchTo(r.path); }));
      });
      var remoteRoots = (info && info.remote_roots) || [];
      remoteWrap.classList.toggle("hidden", !remoteRoots.length);
      remoteList.innerHTML = "";
      remoteRoots.forEach(function (r) {
        remoteList.appendChild(knownItem(r, function () { addRemote(r.path); }));
      });
    });
  }

  function setMode(next) {
    mode = next;
    modeLocalBtn.classList.toggle("active", mode === "local");
    modeRemoteBtn.classList.toggle("active", mode === "remote");
    descEl.textContent = mode === "remote" ? DESC_REMOTE : DESC_LOCAL;
    chooseBtn.textContent = mode === "remote" ? "加入同步" : "选择此目录";
    document.getElementById("root-picker-known").classList.toggle("hidden", mode !== "local");
    document.getElementById("root-picker-remote-known").classList.toggle("hidden", mode !== "remote");
    loadKnownRoots();
    if (mode === "remote") {
      api.getOrNull("/api/remotes").then(function (info) {
        var first = ((info && info.remotes) || []).filter(function (r) { return r.ssh; })[0];
        browse((first && first.remote_dir) || "/");
      });
    } else {
      api.getOrNull("/api/data_root").then(function (info) {
        browse((info && info.data_root) || "/");
      });
    }
  }

  document.getElementById("data-root-btn").addEventListener("click", function () {
    overlay.classList.remove("hidden");
    api.getOrNull("/api/remotes").then(function (info) {
      sshRemotes = ((info && info.remotes) || []).filter(function (r) { return r.ssh; });
      remoteName = sshRemotes.length ? sshRemotes[0].name : null;
      /* 有 ssh 远端时默认远端模式（数据在 H-Server 上），否则只能本机 */
      modeRemoteBtn.classList.toggle("hidden", !sshRemotes.length);
      setMode(sshRemotes.length ? "remote" : "local");
    });
  });
  modeLocalBtn.addEventListener("click", function () { setMode("local"); });
  modeRemoteBtn.addEventListener("click", function () { setMode("remote"); });
  document.getElementById("root-picker-close").addEventListener("click", function () {
    overlay.classList.add("hidden");
  });
  overlay.addEventListener("click", function (ev) {
    if (ev.target === overlay) overlay.classList.add("hidden");
  });
  document.getElementById("root-picker-up").addEventListener("click", function () {
    var url = "/api/browse?path=" + encodeURIComponent(current || "/");
    if (mode === "remote") url += "&remote=" + encodeURIComponent(remoteName || "1");
    api.get(url).then(function (info) {
      if (info.parent) browse(info.parent);
    });
  });
  chooseBtn.addEventListener("click", function () {
    if (!current) return;
    if (mode === "remote") addRemote(current);
    else switchTo(current);
  });
}
