"""
whatsapp_bot/app.py — DECAFIA WhatsApp diagnostic bot.

Stack  : Flask + Twilio API + ONNX Runtime
Model  : decafia_int8.onnx (loaded once at startup)
Memory : under 512 MB RAM (free-tier compatible)

Conversation flow:
  - User sends photo  → download → ONNX inference → diagnosis message
  - User sends text   → command dispatch (TRATAMIENTO, HISTORIAL, AYUDA)

Run locally:
  pip install flask twilio onnxruntime pillow requests
  export TWILIO_AUTH_TOKEN=your_token
  python whatsapp_bot/app.py
"""

import json
import logging
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix
from PIL import Image
from twilio.request_validator import RequestValidator
from twilio.twiml.messaging_response import MessagingResponse

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
MODELS_PATH     = BASE_DIR / "models"
MODEL_PATH      = MODELS_PATH / "decafia_int8.onnx"
TREATMENTS_PATH = BASE_DIR / "treatments.json"

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD         = 0.25
IOU_THRESHOLD          = 0.45
INPUT_SIZE             = 640
CLASS_NAMES            = {0: "roya", 1: "coco", 2: "sano", 3: "minador"}
DISEASE_IDS            = {0, 1, 3}   # sano (2) is healthy — no treatment needed
HISTORY_LIMIT          = 5
TWILIO_FROM_NUMBER     = os.environ.get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+15559505869")
DEBUG_INFERENCE        = os.environ.get("DEBUG_INFERENCE", "false").lower() == "true"

_log_level = logging.DEBUG if DEBUG_INFERENCE else logging.INFO
# basicConfig is a no-op if Flask/Werkzeug already registered handlers, so force
# the level explicitly on both the root logger and this module's logger.
logging.basicConfig(level=_log_level, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().setLevel(_log_level)
log = logging.getLogger(__name__)
log.setLevel(_log_level)

# ── Load model once at startup (module level so gunicorn --preload captures it) ─
try:
    _session    = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    _input_name = _session.get_inputs()[0].name
    log.info("ONNX model loaded: %s", MODEL_PATH)
except Exception as e:
    log.error("Failed to load ONNX model: %s", e)
    _session    = None
    _input_name = None

# ── Load treatments ───────────────────────────────────────────────────────────
with TREATMENTS_PATH.open("r", encoding="utf-8") as f:
    TREATMENTS: dict = json.load(f)

# ── Per-user session history (in-memory, resets on restart) ──────────────────
# { whatsapp_number: [ {timestamp, disease, confidence, area_pct}, ... ] }
_history: dict[str, list] = defaultdict(list)

# ── Low-confidence fallback message ──────────────────────────────────────────
LOW_CONFIDENCE_MESSAGE = (
    "⚠️ *No se detectaron enfermedades con suficiente certeza.*\n\n"
    "Para un mejor resultado:\n"
    "  • Acerca la cámara a la hoja afectada\n"
    "  • Asegúrate de que la hoja ocupe la mayor parte de la foto\n"
    "  • Usa buena iluminación natural\n"
    "  • Enfoca directamente en el síntoma visible"
)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
# Trust Render's reverse-proxy headers so request.url returns the public HTTPS URL.
# Without this, Twilio signature validation fails (http:// vs https:// mismatch).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
print(f"FROM number: {os.environ.get('TWILIO_WHATSAPP_NUMBER', 'NOT SET')}")


# ── Inference helpers ─────────────────────────────────────────────────────────

def preprocess(image: Image.Image) -> tuple[np.ndarray, int, int]:
    """Resize + normalize image to model input. Returns (tensor, orig_w, orig_h)."""
    orig_w, orig_h = image.size
    log.debug("Preprocess — input size: %dx%d (WxH)", orig_w, orig_h)
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # NCHW
    log.debug("Preprocess — tensor shape after resize+transpose: %s", arr.shape)
    return arr, orig_w, orig_h


def nms(boxes, scores, iou_thresh: float) -> list[int]:
    """Simple NMS. boxes: (N,4) xyxy, scores: (N,)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou   = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def run_inference(image: Image.Image) -> list[dict]:
    """
    Run ONNX inference. Returns list of detections:
      [{class_id, class_name, confidence, area_pct}]
    Sorted by confidence descending.
    """
    if _session is None:
        raise RuntimeError("Model not loaded")

    tensor, orig_w, orig_h = preprocess(image)
    outputs = _session.run(None, {_input_name: tensor})
    raw     = outputs[0]
    log.debug("ONNX raw output shape: %s", raw.shape)   # expect (1, 8, 8400)

    pred = raw[0]   # drop batch dim → (8, 8400) or (8400, 8)

    # ultralytics ONNX output: (1, 4+nc, 8400) → after [0]: (4+nc, 8400)
    # rows = features, cols = anchors — transpose to (8400, 4+nc)
    if pred.shape[0] < pred.shape[1]:
        log.debug("Transposing pred from %s → (%d, %d)", pred.shape, pred.shape[1], pred.shape[0])
        pred = pred.T   # → (8400, 4+nc)

    log.debug("pred shape after transpose: %s  (anchors x features)", pred.shape)

    boxes_xywh   = pred[:, :4]       # cx, cy, w, h — in model input space (640x640 pixels)
    class_scores = pred[:, 4:]       # nc columns
    log.debug("class_scores shape: %s  (num_classes=%d)", class_scores.shape, class_scores.shape[1])

    # Coords are in 640x640 input space — scale back to original image pixels
    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0
    cx = boxes_xywh[:, 0] * scale_x
    cy = boxes_xywh[:, 1] * scale_y
    bw = boxes_xywh[:, 2] * scale_x
    bh = boxes_xywh[:, 3] * scale_y
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    best_cls   = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)

    log.debug("Raw anchors: %d  |  max score overall: %.4f  |  top-5 scores: %s",
              len(best_score), best_score.max(),
              sorted(best_score, reverse=True)[:5])

    mask     = best_score >= CONF_THRESHOLD
    boxes_f  = boxes_xyxy[mask]
    scores_f = best_score[mask]
    cls_f    = best_cls[mask]
    log.debug("After conf>=%.2f filter: %d detections", CONF_THRESHOLD, len(scores_f))

    for i in range(len(scores_f)):
        log.debug("  [pre-NMS] cls=%d (%s)  conf=%.4f  box=[%.1f,%.1f,%.1f,%.1f]",
                  int(cls_f[i]), CLASS_NAMES.get(int(cls_f[i]), "?"),
                  scores_f[i], boxes_f[i][0], boxes_f[i][1], boxes_f[i][2], boxes_f[i][3])

    keep = nms(boxes_f, scores_f, IOU_THRESHOLD)
    log.debug("After NMS: %d detections survive", len(keep))

    image_area = orig_w * orig_h
    detections = []
    for idx in keep:
        cls_id = int(cls_f[idx])
        bx1, by1, bx2, by2 = boxes_f[idx]
        box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        area_pct = min(100.0, box_area / image_area * 100)
        log.debug("  [post-NMS] cls=%d (%s)  conf=%.4f  area=%.1f%%  disease=%s",
                  cls_id, CLASS_NAMES.get(cls_id, "?"), scores_f[idx],
                  area_pct, cls_id in DISEASE_IDS)
        if cls_id not in DISEASE_IDS:
            continue
        detections.append({
            "class_id":   cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": float(scores_f[idx]),
            "area_pct":   round(area_pct, 1),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # Count lesion boxes per class before dedup
    from collections import Counter
    lesion_counts: dict[int, int] = Counter(d["class_id"] for d in detections)

    # Keep only the highest-confidence detection per class
    seen: set[int] = set()
    deduped = []
    for d in detections:
        if d["class_id"] not in seen:
            seen.add(d["class_id"])
            d["lesion_count"] = lesion_counts[d["class_id"]]
            deduped.append(d)
    detections = deduped

    # Cap at top 2 to stay within WhatsApp's 1600-char limit
    detections = detections[:2]

    if detections:
        top = detections[0]
        log.info("Final decision: %s (conf=%.4f, lesions=%d)",
                 top["class_name"], top["confidence"], top["lesion_count"])
    else:
        log.info("Final decision: sano (no disease detections above threshold)")
    return detections


# ── Message builders ──────────────────────────────────────────────────────────

def _disease_display(class_name: str) -> str:
    names = {"roya": "Roya", "coco": "Coco — Picudo del Café (Vaquita)", "minador": "Minador de la hoja"}
    return names.get(class_name, class_name.capitalize())


def _urgency_icon(class_name: str) -> str:
    t = TREATMENTS.get(class_name, {})
    return t.get("emoji", "⚠️")


def build_diagnosis_message(detections: list[dict]) -> str:
    if not detections:
        return (
            "✅ *Hoja sana*\n"
            "No se detectaron enfermedades.\n"
            "Revisa nuevamente en 2 semanas.\n\n"
            "Responde *AYUDA* para ver comandos."
        )

    lines = ["🔍 *Diagnóstico DECAFIA*\n"]

    for d in detections:
        icon  = _urgency_icon(d["class_name"])
        name  = _disease_display(d["class_name"])
        conf  = round(d["confidence"] * 100)
        area  = d["area_pct"]
        t     = TREATMENTS.get(d["class_name"], {})
        urg   = t.get("urgencia", "")
        fund  = t.get("fungicida", "")
        dosis = t.get("dosis", "")

        count = d["lesion_count"]
        lines.append(
            f"{icon} *{name} detectada* — confianza: {conf}%\n"
            f"Lesiones detectadas: {count}\n"
        )
        if fund:
            lines.append(
                f"💊 *Recomendación:*\n"
                f"{fund}.\n"
                f"Dosis: {dosis}. {urg}.\n"
            )
        cmd = f"TRATAMIENTO {d['class_name'].upper()}"
        lines.append(f"Responde *{cmd}* para guía completa.\n")

    lines.append("Responde *HISTORIAL* para ver tus últimos escaneos.")
    lines.append("Responde *AYUDA* para ver todos los comandos.")
    return "\n".join(lines)


def build_treatment_message(class_name: str) -> str:
    t = TREATMENTS.get(class_name)
    if not t:
        return f"No encontré información de tratamiento para '{class_name}'."

    steps = "\n".join(f"  • {s}" for s in t.get("medidas_culturales", []))
    return (
        f"{t['emoji']} *{t['nombre']}*\n"
        f"_{t['nombre_cientifico']}_\n\n"
        f"📋 *Descripción:*\n{t['descripcion']}\n\n"
        f"🔬 *Síntomas:*\n{t['sintomas']}\n\n"
        f"💊 *Tratamiento químico:*\n"
        f"Producto: {t['fungicida']}\n"
        f"Dosis: {t['dosis']}\n"
        f"Frecuencia: {t['frecuencia']}\n"
        f"Aplicación: {t['modo_aplicacion']}\n\n"
        f"🌿 *Medidas culturales:*\n{steps}\n\n"
        f"💰 *Costo estimado:* {t['costo_estimado_cop']}\n"
        f"📚 *Fuente:* {t['fuente']}\n\n"
        f"⚠️ *Urgencia:* {t['urgencia']}"
    )


def build_history_message(number: str) -> str:
    records = _history.get(number, [])
    if not records:
        return "📋 *Historial vacío*\nAún no has enviado ninguna foto para analizar."

    lines = [f"📋 *Últimos {len(records)} escaneos:*\n"]
    for r in reversed(records[-HISTORY_LIMIT:]):
        ts      = r["timestamp"].strftime("%d/%m %H:%M")
        disease = _disease_display(r["disease"]) if r["disease"] != "sano" else "Sana"
        conf    = round(r["confidence"] * 100)
        lines.append(f"• {ts} — {disease} ({conf}%)")
    return "\n".join(lines)


HELP_MESSAGE = (
    "🌿 *DECAFIA — Comandos disponibles*\n\n"
    "📸 *Envía una foto* de una hoja de café para analizar.\n\n"
    "💊 *TRATAMIENTO ROYA* — Guía para tratar la roya\n"
    "💊 *TRATAMIENTO COCO* — Guía para tratar el Picudo del Café (Coco/Vaquita)\n"
    "💊 *TRATAMIENTO MINADOR* — Guía para tratar el minador\n"
    "📋 *HISTORIAL* — Tus últimos 5 escaneos\n"
    "❓ *AYUDA* — Ver este mensaje\n\n"
    "_🌿 DECAFIA — Detección Inteligente de Enfermedades del Café._"
)

NOT_A_LEAF_MESSAGE = (
    "⚠️ No pude analizar la imagen.\n"
    "Envía una foto clara de una hoja de café.\n\n"
    "Consejos para mejor diagnóstico:\n"
    "• Foto con buena iluminación natural\n"
    "• Hoja que ocupe la mayor parte de la imagen\n"
    "• Evitar reflejos y sombras fuertes"
)


# ── Image download ────────────────────────────────────────────────────────────

def download_image(media_url: str, auth_token: str) -> Optional[Image.Image]:
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    try:
        resp = requests.get(
            media_url,
            auth=(account_sid, auth_token),
            timeout=15,
        )
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
        return Image.open(tmp_path)
    except Exception as e:
        log.error("Failed to download image %s: %s", media_url, e)
        return None


# ── Webhook ───────────────────────────────────────────────────────────────────

def _twiml(text: str):
    """Return a properly formatted TwiML response with Content-Type text/xml."""
    resp = MessagingResponse()
    msg = resp.message()
    msg.body(text)
    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/webhook", methods=["POST"])
def webhook():
    print("WEBHOOK CALLED", flush=True)
    print(f"Body: {request.form.get('Body', '')}", flush=True)
    print(f"NumMedia: {request.form.get('NumMedia', '0')}", flush=True)
    print(f"From: {request.form.get('From', '')}", flush=True)

    auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")

    # Validate Twilio signature
    if auth_token:
        validator = RequestValidator(auth_token)
        signature = request.headers.get("X-Twilio-Signature", "")
        url       = request.url
        log.info("Validating signature for URL: %s", url)
        if not validator.validate(url, request.form, signature):
            log.warning("Invalid Twilio signature — rejected request (url=%s)", url)
            return "Forbidden", 403

    sender  = request.form.get("From", "unknown")
    body    = (request.form.get("Body") or "").strip().upper()
    n_media = int(request.form.get("NumMedia", 0))

    print(f"Parsed — sender={sender}  body={body!r}  n_media={n_media}", flush=True)

    # ── Photo received ────────────────────────────────────────────────────────
    if n_media > 0:
        media_url = request.form.get("MediaUrl0", "")
        image = download_image(media_url, auth_token)

        if image is None:
            return _twiml(NOT_A_LEAF_MESSAGE)

        try:
            detections = run_inference(image)
        except Exception as e:
            log.error("Inference error: %s", e)
            return _twiml("⚠️ Error al analizar la imagen. Intenta de nuevo.")

        # Save to history
        if detections:
            top = detections[0]
            _history[sender].append({
                "timestamp":  datetime.now(),
                "disease":    top["class_name"],
                "confidence": top["confidence"],
                "lesions":    top["lesion_count"],
            })
        else:
            _history[sender].append({
                "timestamp":  datetime.now(),
                "disease":    "sano",
                "confidence": 1.0,
                "lesions":    0,
            })

        if len(_history[sender]) > HISTORY_LIMIT * 2:
            _history[sender] = _history[sender][-HISTORY_LIMIT:]

        if not detections:
            return _twiml(LOW_CONFIDENCE_MESSAGE)
        return _twiml(build_diagnosis_message(detections))

    # ── Text commands ─────────────────────────────────────────────────────────
    if re.match(r"TRATAMIENTO\s+ROYA", body):
        reply = build_treatment_message("roya")
    elif re.match(r"TRATAMIENTO\s+COCO", body):
        reply = build_treatment_message("coco")
    elif re.match(r"TRATAMIENTO\s+MINADOR", body):
        reply = build_treatment_message("minador")
    elif body == "HISTORIAL":
        reply = build_history_message(sender)
    elif body in ("AYUDA", "HELP", "HOLA", "INICIO", "START"):
        reply = HELP_MESSAGE
    else:
        reply = (
            "No entendí tu mensaje.\n"
            "Responde *AYUDA* para ver los comandos disponibles,\n"
            "o envía una foto de una hoja de café para analizarla."
        )

    print(f"Replying with {len(reply)} chars", flush=True)
    return _twiml(reply)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model_loaded": _session is not None}, 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "service": "DECAFIA Bot"})


@app.route("/", methods=["POST"])
def index_webhook():
    return webhook()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting DECAFIA WhatsApp bot on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
