# DECAFIA WhatsApp Bot

Diagnostic bot for coffee leaf diseases. Send a photo, receive a diagnosis and treatment guide in Spanish.

## Requirements

```
pip install flask twilio onnxruntime pillow requests
```

## Local setup with ngrok

1. **Set environment variables:**
   ```bash
   export TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   export TWILIO_AUTH_TOKEN=your_auth_token
   ```

2. **Start the bot:**
   ```bash
   python whatsapp_bot/app.py
   ```

3. **Expose locally with ngrok:**
   ```bash
   ngrok http 5000
   ```
   Copy the HTTPS URL, e.g. `https://abc123.ngrok.io`

4. **Configure Twilio sandbox:**
   - Go to [Twilio Console](https://console.twilio.com) → Messaging → Try it out → Send a WhatsApp message
   - In "Sandbox Settings", set the webhook to:
     ```
     https://abc123.ngrok.io/webhook
     ```
   - Method: **HTTP POST**

5. **Test:**
   - On your phone, send `join <sandbox-keyword>` to the Twilio sandbox number
   - Send a photo of a coffee leaf
   - Send `AYUDA` for the command list

## Free deployment on Render.com

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo, select the branch
4. **Build command:** `pip install -r requirements.txt`
5. **Start command:** `python whatsapp_bot/app.py`
6. Add environment variables: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`
7. Copy the Render URL and paste it as the Twilio webhook

> **Note:** Free Render instances sleep after 15 minutes of inactivity.
> Use [UptimeRobot](https://uptimerobot.com) to ping `/health` every 10 minutes to keep it awake.

## Commands

| Command | Description |
|---------|-------------|
| (send photo) | Analyze a coffee leaf |
| `TRATAMIENTO ROYA` | Full treatment guide for coffee rust |
| `TRATAMIENTO COCO` | Full treatment guide for mealybug |
| `TRATAMIENTO MINADOR` | Full treatment guide for leaf miner |
| `HISTORIAL` | Last 5 scans this session |
| `AYUDA` | Show all commands |

## Detected diseases

| Class | Spanish name | Description |
|-------|-------------|-------------|
| roya | Roya del cafeto | Coffee rust — orange/yellow spots |
| coco | Coco — Picudo del Café (Vaquita) | Leaf-chewing weevil (Curculionidae: Compsus sp. / Epicaerus sp.) — irregular perforations |
| minador | Minador de la hoja | Leaf miner — translucent galleries |
| sano | Hoja sana | Healthy — no disease detected |
