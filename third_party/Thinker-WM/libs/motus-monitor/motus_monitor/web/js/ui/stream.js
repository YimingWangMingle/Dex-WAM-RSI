/* Motus Monitor 前端 —— 数据集二进制/流式传输协议（dataset 模块复用）。
 * 从旧 web/dataset.js:85-179 逐字迁出。
 *
 * 二进制端点打包格式：4 字节大端 JSON 长度 + JSON 头 + blob（见 server.py _send_packed）。
 * 流式端点（stream=1）：前缀行 PRG <frac> <stage>（服务器解码进度，0..N 条）
 * 或 PRGJ <json>（结构化进度：frac/label/detail/eta_s，切片聚合分阶段用）
 * + LEN <n> + n 字节成品；出错为 ERR <json>。
 */

export function parsePacked(buf) {
  var view = new DataView(buf);
  var jsonLen = view.getUint32(0, false);
  var meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen)));
  return { meta: meta, blob: buf.slice(4 + jsonLen) };
}

export function fetchPacked(url) {
  return fetch(url).then(function (resp) {
    if (!resp.ok) {
      return resp.json().then(function (body) {
        throw new Error(body.error || ("HTTP " + resp.status));
      });
    }
    return resp.arrayBuffer();
  }).then(parsePacked);
}

/* fetchStream handlers:
 *   onServerProgress(frac, stage)  集群解码推进
 *   onServerProgressJson(event)    结构化分阶段进度
 *   onDownload(received, total)    成品下载推进
 * resolve 为成品字节的 ArrayBuffer。 */
export function fetchStream(url, handlers) {
  handlers = handlers || {};
  return fetch(url).then(function (resp) {
    if (!resp.ok) {
      return resp.json().then(function (body) {
        throw new Error(body.error || ("HTTP " + resp.status));
      });
    }
    var reader = resp.body.getReader();
    var pending = new Uint8Array(0);
    var mode = "head";
    var bodyTotal = -1;
    var decoder = new TextDecoder();
    function concat(a, b) {
      var c = new Uint8Array(a.length + b.length);
      c.set(a);
      c.set(b, a.length);
      return c;
    }
    function pump() {
      return reader.read().then(function (res) {
        if (res.value) pending = concat(pending, res.value);
        var i;
        while (mode === "head") {
          var nl = -1;
          for (i = 0; i < pending.length; i++) { if (pending[i] === 10) { nl = i; break; } }
          if (nl < 0) break;
          var line = decoder.decode(pending.slice(0, nl));
          pending = pending.slice(nl + 1);
          if (line.lastIndexOf("PRG ", 0) === 0) {
            var parts = line.slice(4).split(" ");
            if (handlers.onServerProgress) {
              handlers.onServerProgress(parseFloat(parts[0]), parts[1] || "");
            }
          } else if (line.lastIndexOf("PRGJ ", 0) === 0) {
            if (handlers.onServerProgressJson) {
              try { handlers.onServerProgressJson(JSON.parse(line.slice(5))); } catch (e) { }
            }
          } else if (line.lastIndexOf("LEN ", 0) === 0) {
            bodyTotal = parseInt(line.slice(4), 10) || 0;
            mode = "body";
          } else if (line.lastIndexOf("ERR ", 0) === 0) {
            var msg = "preview stream error";
            try { msg = JSON.parse(line.slice(4)).error || msg; } catch (e) { }
            throw new Error(msg);
          }
        }
        if (mode === "body" && handlers.onDownload) {
          handlers.onDownload(pending.length, bodyTotal);
        }
        if (res.done) {
          if (mode !== "body") throw new Error("bad preview stream");
          return pending.buffer.slice(pending.byteOffset, pending.byteOffset + pending.length);
        }
        return pump();
      });
    }
    return pump();
  });
}

export function jpegUrls(meta, blob) {
  var urls = [];
  var pos = 0;
  for (var i = 0; i < meta.sizes.length; i++) {
    urls.push(URL.createObjectURL(new Blob([blob.slice(pos, pos + meta.sizes[i])], { type: "image/jpeg" })));
    pos += meta.sizes[i];
  }
  return urls;
}
