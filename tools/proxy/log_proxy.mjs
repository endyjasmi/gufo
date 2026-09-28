#!/usr/bin/env node
// Logging reverse proxy for debugging the Gufo inference server.
//
// Sits between an OpenAI-compatible client (e.g. the pi coding agent) and the
// Gufo HTTP server, forwarding requests byte-for-byte while tee-ing everything
// to disk: full request headers+body and full response status+headers+body
// (including SSE stream transcripts) under --log-dir, plus an index.jsonl with
// one summary line per request.
//
// Usage:
//   node tools/proxy/log_proxy.mjs [--port 11435] [--target http://127.0.0.1:11434]
//                                  [--log-dir logs/proxy] [--max-body 33554432]
//
// Point the client at --port and use the real API key; the proxy forwards the
// Authorization header untouched.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function parseArgs(argv) {
  const opts = {
    port: 11435,
    target: "http://127.0.0.1:11434",
    logDir: path.join(path.dirname(fileURLToPath(import.meta.url)), "logs"),
    maxBody: 32 * 1024 * 1024,
  };
  for (let i = 2; i < argv.length; i++) {
    const flag = argv[i];
    if (flag === "--port") opts.port = Number(argv[++i]);
    else if (flag === "--target") opts.target = argv[++i];
    else if (flag === "--log-dir") opts.logDir = argv[++i];
    else if (flag === "--max-body") opts.maxBody = Number(argv[++i]);
    else throw new Error(`unknown option: ${flag}`);
  }
  return opts;
}

const opts = parseArgs(process.argv);
const target = new URL(opts.target);
fs.mkdirSync(opts.logDir, { recursive: true });
const indexPath = path.join(opts.logDir, "index.jsonl");
const indexStream = fs.createWriteStream(indexPath, { flags: "a" });

let seq = 0;

function ts() {
  return new Date().toISOString();
}

function summarizeBody(body, contentType) {
  // For JSON bodies keep the raw text; for SSE reassemble a readable transcript.
  if (!contentType || !contentType.includes("text/event-stream")) return body;
  const events = [];
  for (const frame of body.split("\n\n")) {
    const dataLines = frame
      .split("\n")
      .filter((l) => l.startsWith("data: "))
      .map((l) => l.slice(6));
    if (dataLines.length === 0) continue;
    const data = dataLines.join("\n");
    if (data === "[DONE]") {
      events.push({ done: true });
      continue;
    }
    try {
      const obj = JSON.parse(data);
      const choice = obj.choices && obj.choices[0];
      const delta = choice && choice.delta;
      events.push({
        finish_reason: choice ? choice.finish_reason : undefined,
        content: delta ? delta.content : undefined,
        reasoning: delta ? (delta.reasoning_content ?? delta.reasoning) : undefined,
        tool_calls: delta ? delta.tool_calls : undefined,
        usage: obj.usage,
        error: obj.error,
      });
    } catch {
      events.push({ raw: data.slice(0, 2000) });
    }
  }
  return { eventCount: events.length, events };
}

const server = http.createServer((req, res) => {
  const id = ++seq;
  const started = Date.now();
  const runDir = path.join(
    opts.logDir,
    `${String(id).padStart(5, "0")}_${started}`,
  );
  fs.mkdirSync(runDir, { recursive: true });

  const chunks = [];
  let truncated = false;
  let total = 0;
  req.on("data", (c) => {
    total += c.length;
    if (total <= opts.maxBody) chunks.push(c);
    else truncated = true;
  });
  req.on("error", () => {});

  req.on("end", () => {
    const body = Buffer.concat(chunks);
    const reqPath = `${req.url}`;
    const reqMeta = {
      id,
      time: new Date(started).toISOString(),
      method: req.method,
      path: reqPath,
      headers: req.headers,
      bodyBytes: body.length,
      bodyTruncated: truncated,
    };
    fs.writeFileSync(
      path.join(runDir, "request.json"),
      JSON.stringify({ ...reqMeta, body: body.toString("utf8") }, null, 2),
    );

    const bodyStr = body.toString("utf8");
    let note = "";
    if (req.method === "POST" && bodyStr) {
      try {
        const parsed = JSON.parse(bodyStr);
        const msgs = parsed.messages || parsed.input;
        note = ` model=${parsed.model} stream=${parsed.stream} messages=${Array.isArray(msgs) ? msgs.length : "?"} bytes=${body.length}`;
      } catch {
        note = " <unparseable JSON body!>";
      }
    }
    console.log(
      `[${ts()}] #${id} -> ${req.method} ${reqPath}${note}`,
    );

    const headers = { ...req.headers };
    delete headers.host;
    delete headers.connection;
    delete headers["content-length"];
    delete headers["transfer-encoding"];
    if (body.length) headers["content-length"] = String(body.length);

    const upstream = http.request(
      {
        protocol: target.protocol,
        hostname: target.hostname,
        port: target.port,
        method: req.method,
        path: reqPath,
        headers,
      },
      (upres) => {
        const respChunks = [];
        let respBytes = 0;
        const contentType = upres.headers["content-type"] || "";
        const isStream = contentType.includes("text/event-stream");
        res.writeHead(upres.statusCode, upres.headers);

        fs.writeFileSync(
          path.join(runDir, "response.meta.json"),
          JSON.stringify(
            { status: upres.statusCode, headers: upres.headers, stream: isStream },
            null,
            2,
          ),
        );

        upres.on("data", (c) => {
          respBytes += c.length;
          if (respBytes <= opts.maxBody) respChunks.push(c);
          res.write(c);
        });
        upres.on("end", () => {
          const dur = Date.now() - started;
          const respBody = Buffer.concat(respChunks).toString("utf8");
          fs.writeFileSync(
            path.join(runDir, "response.json"),
            JSON.stringify(
              {
                ...reqMeta,
                status: upres.statusCode,
                durationMs: dur,
                responseBytes: respBytes,
                responseHeaders: upres.headers,
                body: summarizeBody(respBody, contentType),
              },
              null,
              2,
            ),
          );
          indexStream.write(
            JSON.stringify({
              id,
              time: reqMeta.time,
              method: req.method,
              path: reqPath.split("?")[0],
              reqBytes: body.length,
              status: upres.statusCode,
              respBytes,
              ms: dur,
            }) + "\n",
          );
          const flag =
            upres.statusCode >= 400 ? "  !! ERROR" : isStream ? " (sse)" : "";
          let errNote = "";
          if (upres.statusCode >= 400) {
            try {
              const e = JSON.parse(respBody).error;
              errNote = ` code=${e && e.code} msg=${String(e && e.message).slice(0, 300)}`;
            } catch {
              errNote = ` body=${respBody.slice(0, 200)}`;
            }
          }
          console.log(
            `[${ts()}] #${id} <- ${upres.statusCode} ${respBytes}B ${dur}ms${flag}${errNote}`,
          );
          res.end();
        });
        upres.on("error", (e) => {
          console.error(`[${ts()}] #${id} upstream error: ${e.message}`);
          fs.writeFileSync(
            path.join(runDir, "upstream_error.json"),
            JSON.stringify({ error: e.message, code: e.code }),
          );
          res.destroy(e);
        });
      },
    );

    upstream.on("error", (e) => {
      const dur = Date.now() - started;
      console.error(`[${ts()}] #${id} connect error: ${e.code} ${e.message}`);
      indexStream.write(
        JSON.stringify({
          id,
          time: reqMeta.time,
          method: req.method,
          path: reqPath.split("?")[0],
          reqBytes: body.length,
          status: 502,
          error: e.message,
          ms: dur,
        }) + "\n",
      );
      if (!res.headersSent) {
        res.writeHead(502, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: e.message, code: e.code } }));
      } else {
        res.destroy();
      }
    });

    upstream.end(body);
  });

  const abort = () => {
    console.log(`[${ts()}] #${id} client aborted`);
    fs.writeFileSync(
      path.join(runDir, "client_aborted.json"),
      JSON.stringify({ time: ts() }),
    );
    req.destroy();
  };
  res.on("close", () => {
    if (!res.writableEnded) abort();
  });
});

server.listen(opts.port, "127.0.0.1", () => {
  console.log(
    `[${ts()}] log proxy listening on http://127.0.0.1:${opts.port} -> ${opts.target}, logging to ${opts.logDir}`,
  );
});
