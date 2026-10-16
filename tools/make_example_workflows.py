"""Build example_workflows/*.json with the real ComfyUI frontend (headless Chrome, DevTools protocol).

The graphs are created through LiteGraph in a running ComfyUI that has this node pack installed,
then serialized by the frontend itself, so slot order and widget values always match what
ComfyUI expects. Each workflow is also converted to an API prompt as a check, and the README screenshots are taken in docs/.

usage: python tools/make_example_workflows.py http://127.0.0.1:8188 [--chrome google-chrome] [--queue]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "example_workflows"

SCREENSHOTS = {"drama3_small_voice_generation": "workflow_generate_voice.png",
               "drama3_small_keep_generated_voice": "workflow_keep_voice.png"}
REFERENCE_TRANSCRIPT = "Replace with the exact words spoken in your reference clip."

# Each workflow: nodes (id, type, pos, widget values by name) and links (from, slot, to, input name).
WORKFLOWS = {
    "drama3_small_voice_generation": {
        "nodes": [
            (1, "Drama3SmallLoader", [40, 140], {}),
            (2, "Drama3SmallTTS", [420, 80], {
                "text": "The old lighthouse had been dark for nearly forty years. [sighs] Nobody ever went to check.",
                "voice_description": "a warm, steady male narrator's voice", "seed": 7}),
            (3, "PreviewAudio", [900, 120], {}),
        ],
        "links": [(1, 0, 2, "model"), (2, 0, 3, "audio")],
    },
    "drama3_small_keep_generated_voice": {
        "nodes": [
            (1, "Drama3SmallLoader", [40, 300], {}),
            (2, "Drama3SmallTTS", [420, 40], {
                "text": "今日はとても良い天気ですね。散歩にでも行きましょうか。", "voice_description": "", "seed": 21}),
            (4, "Drama3SmallTTS", [900, 300], {
                "text": "それでは、次の議題に移りましょう。資料の三ページ目をご覧ください。", "voice_description": "", "seed": 5}),
            (3, "PreviewAudio", [900, 40], {}),
            (5, "PreviewAudio", [1380, 300], {}),
        ],
        "links": [(1, 0, 2, "model"), (1, 0, 4, "model"), (2, 0, 4, "reference_audio"), (2, 1, 4, "reference_text"),
                  (2, 0, 3, "audio"), (4, 0, 5, "audio")],
    },
    "drama3_small_voice_cloning": {
        "nodes": [
            (1, "Drama3SmallLoader", [40, 60], {}),
            (6, "LoadAudio", [40, 260], {"audio": "your_reference.wav"}),
            (2, "Drama3SmallTTS", [420, 60], {
                "text": "[laughs] まさかそんなオチだとは思わなかった。[sighs] でも、ちょっと安心したよ。",
                "voice_description": "", "seed": 3, "reference_text": REFERENCE_TRANSCRIPT}),
            (3, "PreviewAudio", [900, 60], {}),
        ],
        "links": [(1, 0, 2, "model"), (6, 0, 2, "reference_audio"), (2, 0, 3, "audio")],
    },
}

BUILD_JS = r"""
async (spec) => {
  const app = window.comfyAPI?.app?.app ?? window.app;
  app.graph.clear();
  const byId = {};
  for (const [id, type, pos, values] of spec.nodes) {
    const node = LiteGraph.createNode(type);
    if (!node) throw new Error("unknown node " + type);
    node.id = id;
    node.pos = pos;
    app.graph.add(node);
    for (const [name, value] of Object.entries(values)) {
      const w = node.widgets?.find((w) => w.name === name);
      if (!w) throw new Error(type + " has no widget " + name);
      w.value = value;
    }
    const seed = node.widgets?.find((w) => w.name === "control_after_generate");
    if (seed) seed.value = "fixed";
    byId[id] = node;
  }
  for (const [from, slot, to, input] of spec.links) {
    const target = byId[to];
    const index = target.inputs.findIndex((i) => i.name === input);
    if (index < 0) throw new Error(target.type + " has no input " + input);
    byId[from].connect(slot, target, index);
  }
  app.graph.setDirtyCanvas(true, true);
  const workflow = app.graph.serialize();
  const prompt = (await app.graphToPrompt()).output;
  return JSON.stringify({ workflow, prompt });
}
"""


FIT_JS = r"""
(async () => {
  const app = window.comfyAPI?.app?.app ?? window.app;
  try { await app.extensionManager.command.execute("Comfy.Canvas.FitView"); }
  catch (e) { app.canvas.ds.offset = [40, 40]; app.canvas.ds.scale = 0.8; }
  app.graph.setDirtyCanvas(true, true);
  return true;
})()
"""


async def cdp(ws, method, params=None, _id=[0]):
    _id[0] += 1
    await ws.send_json({"id": _id[0], "method": method, "params": params or {}})
    while True:
        msg = await ws.receive_json()
        if msg.get("id") == _id[0]:
            if "error" in msg:
                raise RuntimeError(msg["error"])
            return msg["result"]


async def main(base: str, chrome: str, queue: bool) -> None:
    profile = tempfile.mkdtemp(prefix="drama3_wf_chrome_")
    port = 9333
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--window-size=1600,1000", base],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(60):
                try:
                    async with session.get(f"http://127.0.0.1:{port}/json") as r:
                        pages = [p for p in await r.json() if p["type"] == "page"]
                    if pages:
                        break
                except aiohttp.ClientError:
                    pass
                time.sleep(0.5)
            async with session.ws_connect(pages[0]["webSocketDebuggerUrl"], max_msg_size=0) as ws:
                for _ in range(120):  # wait for the app and our node definitions
                    res = await cdp(ws, "Runtime.evaluate", {"expression":
                        "!!(window.LiteGraph && LiteGraph.registered_node_types['Drama3SmallTTS'] && (window.comfyAPI?.app?.app ?? window.app)?.graph)",
                        "returnByValue": True})
                    if res["result"].get("value"):
                        break
                    time.sleep(1)
                else:
                    raise RuntimeError("ComfyUI frontend did not load the drama3-small nodes")
                OUT.mkdir(exist_ok=True)
                for name, spec in WORKFLOWS.items():
                    res = await cdp(ws, "Runtime.evaluate", {
                        "expression": f"({BUILD_JS})({json.dumps(spec, ensure_ascii=False)})",
                        "awaitPromise": True, "returnByValue": True})
                    if "exceptionDetails" in res:
                        raise RuntimeError(f"{name}: {res['exceptionDetails']}")
                    data = json.loads(res["result"]["value"])
                    (OUT / f"{name}.json").write_text(json.dumps(data["workflow"], ensure_ascii=False, indent=2) + "\n")
                    prompt = data["prompt"]
                    print(f"{name}: {len(data['workflow']['nodes'])} nodes, {len(data['workflow']['links'])} links; prompt nodes "
                          + ", ".join(f"{k}:{v['class_type']}" for k, v in prompt.items()), flush=True)
                    # Close first-run dialogs (Templates) and fit the graph into view.
                    for _ in range(2):
                        for kind in ("keyDown", "keyUp"):
                            await cdp(ws, "Input.dispatchKeyEvent", {"type": kind, "key": "Escape", "code": "Escape",
                                                                    "windowsVirtualKeyCode": 27})
                    await cdp(ws, "Runtime.evaluate", {"expression": FIT_JS, "awaitPromise": True})
                    time.sleep(1.5)
                    if name in SCREENSHOTS:
                        shot = await cdp(ws, "Page.captureScreenshot", {"format": "png"})
                        (HERE / "docs" / SCREENSHOTS[name]).write_bytes(base64.b64decode(shot["data"]))
                    if queue:
                        async with session.post(f"{base}/prompt", json={"prompt": prompt}) as r:
                            print("  queued:", (await r.json()).get("prompt_id"))
    finally:
        proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8188")
    ap.add_argument("--chrome", default="google-chrome")
    ap.add_argument("--queue", action="store_true", help="also queue each converted prompt on the server")
    args = ap.parse_args()
    asyncio.run(main(args.base.rstrip("/"), args.chrome, args.queue))
