#!/usr/bin/env node

import { spawn, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, request as httpRequest } from "node:http";
import { resolve } from "node:path";

const [cameraId, outputDir, clipPath, viewerUrl] = process.argv.slice(2);
const token = process.env.CANARY_RELAY_TOKEN;
if (!cameraId || !outputDir || !clipPath || !viewerUrl || !token) {
  console.error("usage: CANARY_RELAY_TOKEN=<token> node deepstream_canary_browser.mjs <camera-id> <output-dir> <clip> <viewer-url>");
  process.exit(2);
}

const screenshot = resolve(outputDir, `viewer-${cameraId}.png`);
const chromium = process.env.CHROMIUM ?? "chromium";
const viewer = viewerUrl;
const deadline = Date.now() + 60_000;
const attach = () => new Promise((resolveAttach, rejectAttach) => {
  const consumer = spawn(
    "curl",
    ["--silent", "--show-error", "--max-time", "30", "--header", `X-Edge-Relay-Token: ${token}`, viewer],
    { stdio: ["ignore", "pipe", "pipe"] },
  );
  let received = Buffer.alloc(0);
  let resolved = false;
  consumer.once("error", rejectAttach);
  consumer.once("close", (code) => {
    if (!resolved) rejectAttach(new Error(`MJPEG consumer exited before a frame: ${code}`));
  });
  consumer.stdout.on("data", (chunk) => {
    if (resolved) return;
    received = Buffer.concat([received, chunk]);
    const start = received.indexOf(Buffer.from([0xff, 0xd8]));
    const end = received.indexOf(Buffer.from([0xff, 0xd9]), Math.max(0, start + 2));
    if (start >= 0 && end > start) {
      resolved = true;
      resolveAttach({ consumer, frame: received.subarray(start, end + 2) });
    }
  });
});

let attachment;
while (!attachment && Date.now() < deadline) {
  try {
    attachment = await attach();
  } catch (error) {
    if (Date.now() >= deadline) {
      console.error(String(error));
      process.exit(1);
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
  }
}
if (!attachment) {
  console.error("MJPEG frame deadline exceeded");
  process.exit(1);
}
const { consumer, frame } = attachment;

let upstreamStatus = null;
let frameDelivered = false;
const proxy = createServer((request, response) => {
  if (request.url !== `/stream/${cameraId}`) {
    response.writeHead(404);
    response.end();
    return;
  }
  const upstream = httpRequest(
    new URL(viewer),
    { headers: { "X-Edge-Relay-Token": token } },
    (upstreamResponse) => {
      upstreamStatus = upstreamResponse.statusCode ?? null;
      if (upstreamStatus !== 200) {
        response.writeHead(upstreamStatus ?? 502, upstreamResponse.headers);
        upstreamResponse.pipe(response);
        return;
      }
      let received = Buffer.alloc(0);
      let sent = false;
      upstreamResponse.on("data", (chunk) => {
        if (sent) return;
        received = Buffer.concat([received, chunk]);
        const start = received.indexOf(Buffer.from([0xff, 0xd8]));
        const end = received.indexOf(
          Buffer.from([0xff, 0xd9]),
          Math.max(0, start + 2),
        );
        if (start < 0 || end <= start) return;
        sent = true;
        const authenticatedFrame = received.subarray(start, end + 2);
        response.writeHead(200, {
          "content-type": "image/jpeg",
          "content-length": authenticatedFrame.length,
        });
        response.end(authenticatedFrame, () => {
          frameDelivered = true;
          upstream.destroy();
        });
      });
      upstreamResponse.once("end", () => {
        if (!sent) response.destroy(new Error("authenticated MJPEG frame missing"));
      });
    },
  );
  upstream.once("error", (error) => response.destroy(error));
  upstream.end();
});
await new Promise((resolveListen, rejectListen) => {
  proxy.once("error", rejectListen);
  proxy.listen(0, "127.0.0.1", resolveListen);
});
const proxyAddress = proxy.address();
if (!proxyAddress || typeof proxyAddress === "string") {
  console.error("browser proxy did not bind an IPv4 port");
  process.exit(1);
}
const browserViewer = `http://127.0.0.1:${proxyAddress.port}/stream/${cameraId}`;
const browserExit = await new Promise((resolveExit) => {
  const browser = spawn(
    chromium,
    [
      "--headless=new",
      "--no-sandbox",
      "--run-all-compositor-stages-before-draw",
      "--virtual-time-budget=5000",
      `--screenshot=${screenshot}`,
      "--window-size=1280,720",
      browserViewer,
    ],
    { stdio: "ignore" },
  );
  let settled = false;
  const finish = (status) => {
    if (settled) return;
    settled = true;
    clearTimeout(timeout);
    resolveExit(status);
  };
  const timeout = setTimeout(() => {
    browser.kill("SIGKILL");
    finish(null);
  }, 30_000);
  browser.once("error", () => finish(null));
  browser.once("close", (status) => finish(status));
});
consumer.kill("SIGTERM");
proxy.closeAllConnections();
await new Promise((resolveClose) => proxy.close(resolveClose));

const probe = spawnSync(
  "ffprobe",
  ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", clipPath],
  { encoding: "utf8", timeout: 15_000 },
);
const clip = readFileSync(clipPath);
const viewerOk =
  browserExit === 0 &&
  upstreamStatus === 200 &&
  frameDelivered &&
  existsSync(screenshot);
const receipt = {
  schema_version: 1,
  camera_id: cameraId,
  mjpeg_frame_sha256: createHash("sha256").update(frame).digest("hex"),
  viewer_ok: viewerOk,
  viewer_exit: browserExit,
  viewer_http_status: upstreamStatus,
  screenshot_sha256: viewerOk ? createHash("sha256").update(readFileSync(screenshot)).digest("hex") : null,
  derivative_playable: probe.status === 0 && probe.stdout.trim().length > 0,
  derivative_sha256: createHash("sha256").update(clip).digest("hex"),
};
writeFileSync(resolve(outputDir, `browser-${cameraId}.json`), `${JSON.stringify(receipt)}\n`, { flag: "wx", mode: 0o400 });
console.log(JSON.stringify(receipt));
process.exit(receipt.viewer_ok && receipt.derivative_playable ? 0 : 1);
