/**
 * DREZZDON AI Vision Studio — Secure Multimodal Backend Proxy (Node.js / Express)
 * Хранит GEMINI_API_KEY исключительно на сервере (никогда не передаёт ключ в клиентский код).
 * Вызывает мультимодальную модель Gemini 2.5 Flash для глубокого понимания сцены,
 * точных координат интересных деталей и генерации осмысленных подписей в стиле DREZZDON.
 */

const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '25mb' }));

const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

const SYSTEM_PROMPT = `You are a Senior Computer Vision & Aesthetic Composition Engine specialized in the DREZZDON visual art style (aesthetic red graphic annotations, bounding boxes, arrows, dashed perspective contours, orthogonal connectors, and evocative short captions).

Analyze the user's photograph independently and return strictly valid JSON matching this schema:
{
  "scene_type": "sky_signal | urban_silhouette | vehicle_street | brutalist_architecture | portrait | nature_landscape | object_detail | interior",
  "dominant_mood": "short description of lighting and atmosphere",
  "overall_confidence": 0.94,
  "headline_caption": {
    "text_line_1": "short evocative phrase (2-4 words)",
    "text_line_2": "italic serif continuation (1-3 words)",
    "anchor_box": [ymin, xmin, ymax, xmax] // normalized 0..1000 in open low-clutter area (e.g. sky or wall)
  },
  "detections": [
    {
      "id": "obj_1",
      "label": "meaningful specific detail (not generic 'car', e.g. 'amber halogen lens' or 'red signal phase')",
      "caption": "evocative or technical DREZZDON caption (e.g. 'i will find you.' or 'the system feels you <3' or 'static equilibrium')",
      "confidence": 0.92,
      "box_2d": [ymin, xmin, ymax, xmax], // normalized 0..1000
      "graphic_type": "box_with_arrow | bracket_frame | dashed_contour | center_glyph | technical_tag | puddle_outline",
      "preferred_label_side": "bottom | top | left | right"
    }
  ],
  "connections": [
    {
      "from_id": "obj_1",
      "to_id": "obj_2",
      "style": "orthogonal_arrow | dashed_line",
      "label": "optional micro-tag"
    }
  ],
  "perspective_lines": [
    {
      "points": [[x1, y1], [x2, y2], [x3, y3]], // normalized 0..1000 along architectural ledges, curbs, cables
      "style": "dashed_red"
    }
  ]
}

Rules:
1. NEVER invent invisible facts or causes. Only annotate genuinely visible details with confidence >= 0.68.
2. Do not label an entire car 'Car' — instead highlight specific visual focal points: headlights, spoiler edge, streetlamp halo, wet asphalt reflection, silhouette head/hands, traffic signal lens, cloud cumulus crown, window glow.
3. Vary composition when variant_seed > 0.`;

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'DREZZDON Multimodal Vision Backend',
    model: GEMINI_MODEL,
    api_configured: Boolean(GEMINI_API_KEY)
  });
});

app.post('/api/analyze', async (req, res) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg', variantSeed = 0, langMode = 'drezzdon_en' } = req.body;
    if (!imageBase64) {
      return res.status(400).json({ error: 'Missing imageBase64 payload' });
    }
    if (!GEMINI_API_KEY) {
      return res.status(503).json({
        error: 'GEMINI_API_KEY is not configured on backend. Falling back to On-Device CV + TensorFlow.js pipeline.'
      });
    }

    const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z+]+;base64,/, '');
    const langInstruction = langMode === 'ru'
      ? 'Write all captions and headline in expressive Russian (or bilingual technical tags).'
      : 'Write captions in authentic DREZZDON lowercase English aesthetic (with optional technical tags).';

    const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
        contents: [{
          role: 'user',
          parts: [
            { inlineData: { mimeType, data: cleanBase64 } },
            { text: `Analyze this photograph for DREZZDON red graphic composition. Variant seed: ${variantSeed}. ${langInstruction}` }
          ]
        }],
        generationConfig: {
          temperature: 0.4 + (variantSeed % 4) * 0.12,
          responseMimeType: 'application/json'
        }
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({ error: `Upstream VLM error: ${errText}` });
    }

    const result = await response.json();
    const rawText = result?.candidates?.[0]?.content?.parts?.[0]?.text || '{}';
    const parsed = JSON.parse(rawText);
    return res.json({ source: 'cloud_gemini_vlm', model: GEMINI_MODEL, analysis: parsed });
  } catch (err) {
    console.error('Analysis error:', err);
    return res.status(500).json({ error: err.message || 'Internal Server Error' });
  }
});

app.listen(PORT, () => {
  console.log(`[DREZZDON Backend] Listening on port ${PORT} (Model: ${GEMINI_MODEL})`);
});
