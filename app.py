import os
import io
import re
import types
import uuid
import secrets
import base64
import zipfile
import traceback
import requests
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, jsonify, redirect,
                   url_for, send_file, flash, session, abort)
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image, ImageDraw
import replicate
import pyembroidery
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# APP CONFIG
# ============================================================
import os
# Render को templates और static फोल्डर का पक्का रास्ता दिखाना
base_dir = os.path.abspath(os.path.dirname(__file__))


# फाइल नामों में छुपे अदृश्य अक्षर (\u200b) अपने आप हटाना
def _clean_names(root):
    for folder, dirs, files in os.walk(root, topdown=False):
        for n in files + dirs:
            clean = n.replace('\u200b', '').replace('\ufeff', '').strip()
            if clean != n:
                try:
                    os.rename(os.path.join(folder, n), os.path.join(folder, clean))
                except Exception as e:
                    print('rename failed:', repr(n), e)


_clean_names(os.path.join(base_dir, 'templates'))
_clean_names(os.path.join(base_dir, 'static'))

app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'surat-embroidery-secret-key-change-me')

# डेटाबेस URL सेट करना (Render PostgreSQL के लिए)
db_url = os.getenv('DATABASE_URL', 'sqlite:///./database.db').strip()

# यहाँ सुधार किया गया है (सिर्फ postgresql:// रखा है ताकि psycopg2 सही से काम करे)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'generated')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Replicate API Token (env variable से लोड करें)
REPLICATE_TOKEN = os.getenv('REPLICATE_API_TOKEN', '')
if REPLICATE_TOKEN:
    os.environ['REPLICATE_API_TOKEN'] = REPLICATE_TOKEN

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'कृपया पहले लॉगिन करें।'

# ============================================================
# SUBSCRIPTION PLANS
# ============================================================
PLANS = {
    'starter':  {'name': 'Starter',  'price': 1999, 'credits': 50,  'dst': True,  'hd': False},
    'pro':      {'name': 'Pro',      'price': 3999, 'credits': 150, 'dst': True,  'hd': True},
    'business': {'name': 'Business', 'price': 6999, 'credits': 300, 'dst': True,  'hd': True},
}

# आपके मोबाइल नंबर (WhatsApp के लिए +91 के साथ)
ADMIN_PHONE_1 = "919793400647" 
ADMIN_PHONE_2 = "918840393880"

# ============================================================
# DATABASE MODELS
# ============================================================
class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    plan          = db.Column(db.String(20), default='trial')
    credits       = db.Column(db.Integer, default=20) # 20 Free Trial Credits
    is_admin      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    
    # ट्रायल और पेमेंट के लिए नए कॉलम्स
    trial_start_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_paid_user     = db.Column(db.Boolean, default=False)

    designs = db.relationship('Design', backref='owner', lazy=True)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)
    
    def is_trial_active(self):
        if self.is_paid_user or self.is_admin:
            return True
        days_passed = (datetime.utcnow() - self.trial_start_date).days
        return days_passed < 15 # 15 दिन का ट्रायल


class Design(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prompt     = db.Column(db.Text, nullable=False)
    image_file = db.Column(db.String(255))
    dst_file   = db.Column(db.String(255))
    svg_file   = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PaymentRequest(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_key   = db.Column(db.String(20), nullable=False)
    amount     = db.Column(db.Integer, nullable=False)
    utr_number = db.Column(db.String(50), nullable=False)
    status     = db.Column(db.String(20), default='pending') # pending, approved, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='payments')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================
# HELPER: AI IMAGE → मशीन-रेडी .DST (सुरक्षित फिल) + असली वेक्टर SVG
# ============================================================
# DST में 1 unit = 0.1 mm
# pyembroidery के कुछ वर्शन में EmbPattern/EmbThread नाम "मॉड्यूल" बन जाते हैं
# ("'module' object is not callable" एरर), इसलिए क्लास सीधे उसकी अपनी फाइल से लेते हैं।
try:
    from pyembroidery.EmbPattern import EmbPattern as _EmbPattern
    from pyembroidery.EmbThread import EmbThread as _EmbThread
except Exception:
    _EmbPattern = pyembroidery.EmbPattern
    _EmbThread = pyembroidery.EmbThread


def _make_thread(r, g, b):
    """धागे का रंग (R, G, B) बनाता है।"""
    thread = _EmbThread()
    try:
        thread.set_color(r, g, b)
    except AttributeError:
        thread.color = (r << 16) | (g << 8) | b
    return thread

try:
    DESIGN_SIZE_MM = max(10.0, min(300.0, float(os.getenv('DESIGN_SIZE_MM', '100'))))
except ValueError:
    DESIGN_SIZE_MM = 100.0      # डिज़ाइन की सबसे लंबी साइड (mm) - Render Environment से बदल सकते हैं

FILL_SPACING    = 4        # लाइनों के बीच की दूरी = 0.4 mm
FILL_STITCH_LEN = 35       # एक टाँका = 3.5 mm
MIN_RUN         = 4        # 0.4 mm से छोटी लाइन नहीं सिली जाती
MIN_AREA        = 150      # 1.5 mm² से छोटे धब्बे हटा दिए जाते हैं
MAX_CONNECT     = 60       # डिज़ाइन के अंदर ही जोड़ने की अधिकतम दूरी = 6 mm
MAX_STITCHES    = 100000   # इससे ज़्यादा टाँके = डिज़ाइन बहुत भारी


def _design_mask(image_path):
    """
    इमेज से सिलने वाला हिस्सा (सफ़ेद=255) निकालता है।
    बैकग्राउंड काला हो या सफ़ेद, अपने आप पहचान लेता है।
    नतीजा असली साइज़ में होता है: 1 पिक्सल = 0.1 mm।
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("इमेज लोड नहीं हो पाई!")

    h, w = img.shape
    k = 1024.0 / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # अपने आप सही थ्रेशोल्ड (Otsu): चमकीला हिस्सा = 255
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # किनारे की पट्टी से बैकग्राउंड पहचानें; बैकग्राउंड सफ़ेद हो तो उल्टा करें
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला")

    # डिज़ाइन के चारों ओर की खाली जगह काटें, फिर असली साइज़ (mm) में लाएँ
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = mask.shape
    s = (DESIGN_SIZE_MM * 10.0) / max(ch, cw)
    mask = cv2.resize(mask, (max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                      interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 0.7 mm से पतली चीज़ें (जो साफ़ सिली नहीं जा सकतीं) और छोटे धब्बे हटाएँ
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.copyMakeBorder(mask, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)   # किनारे की पतली चीज़ें भी हटें
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)[8:-8, 8:-8]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean[labels == i] = 255
    if not clean.any():
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला")
    return clean


def _row_runs(row):
    """एक पंक्ति में लगातार सफ़ेद हिस्सों की (शुरू, अंत) सूची।"""
    padded = np.concatenate(([0], (row > 0).astype(np.int8), [0]))
    d = np.diff(padded)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _inside(mask, x0, y0, x1, y1):
    """क्या दो बिंदुओं के बीच की सीधी लाइन पूरी तरह डिज़ाइन के अंदर है?"""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n).round().astype(int)
    ys = np.linspace(y0, y1, n).round().astype(int)
    h, w = mask.shape
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= w or ys.max() >= h:
        return False
    return bool(np.all(mask[ys, xs] > 0))


def _component_chains(cm):
    """
    आकार को "पट्टियों" (chains) में बाँटता है: हर पट्टी में लगातार पंक्तियाँ जो ऊपर-नीचे जुड़ी हैं।
    इससे छेद या खाली जगह के आर-पार धागा नहीं जाता और ट्रिम कम लगते हैं।
    हर पंक्ति: (बायाँ_x, दायाँ_x, y, पंक्ति_नंबर)
    """
    ys, _ = np.where(cm > 0)
    top, bot = int(ys.min()), int(ys.max())
    chains = []
    active = []
    for idx, y in enumerate(range(top, bot + 1, FILL_SPACING)):
        runs = [(a, b) for a, b in _row_runs(cm[y]) if b - a >= MIN_RUN]
        new_active = []
        used = set()
        for a, b in runs:
            home = None
            for ci in active:
                last = chains[ci][-1]
                if ci not in used and last[3] == idx - 1 and last[0] <= b and a <= last[1]:
                    home = ci
                    break
            if home is None:
                chains.append([])
                home = len(chains) - 1
            used.add(home)
            chains[home].append((a, b, y, idx))
            new_active.append(home)
        active = new_active
    return chains


class _Sewer:
    """टाँके लगाने वाला सहायक: लॉक-स्टिच, ट्रिम, छोटे टाँके, गिनती।"""

    def __init__(self, pattern, cx, cy):
        self.p = pattern
        self.cx = cx
        self.cy = cy
        self.count = 0
        self.x = cx
        self.y = cy

    def _pos(self, x, y):
        return int(round(x - self.cx)), int(round(y - self.cy))

    def stitch(self, x, y):
        X, Y = self._pos(x, y)
        self.p.add_stitch_absolute(pyembroidery.STITCH, X, Y)
        self.x, self.y = x, y
        self.count += 1
        if self.count > MAX_STITCHES:
            raise ValueError("डिज़ाइन बहुत भारी/जटिल है, कृपया सरल प्रॉम्प्ट लिखें।")

    def jump(self, x, y):
        X, Y = self._pos(x, y)
        self.p.add_stitch_absolute(pyembroidery.JUMP, X, Y)
        self.x, self.y = x, y

    def trim(self):
        X, Y = self._pos(self.x, self.y)
        self.p.add_stitch_absolute(pyembroidery.TRIM, X, Y)

    def end(self):
        X, Y = self._pos(self.x, self.y)
        self.p.add_stitch_absolute(pyembroidery.END, X, Y)

    def lock(self, x, y, sign):
        # 4 छोटे टाँके (0.3 mm) ताकि धागा खुले नहीं
        for dx in (3, 0, 3, 0):
            self.stitch(x + sign * dx, y)

    def connect(self, x, y):
        # डिज़ाइन के अंदर ही अगली लाइन तक छोटे टाँकों से जाएँ
        dx, dy = x - self.x, y - self.y
        dist = (dx * dx + dy * dy) ** 0.5
        steps = max(1, int(np.ceil(dist / FILL_STITCH_LEN)))
        x0, y0 = self.x, self.y
        for i in range(1, steps + 1):
            self.stitch(x0 + dx * i / steps, y0 + dy * i / steps)

    def sew_line(self, xs, xe, y, offset):
        # हर पंक्ति में सुई की जगह थोड़ी खिसकाते हैं ताकि फिल में लकीरें न बनें
        length = abs(xe - xs)
        d = 1 if xe >= xs else -1
        pos = offset if offset > 0 else FILL_STITCH_LEN
        while pos < length - 8:
            self.stitch(xs + d * pos, y)
            pos += FILL_STITCH_LEN
        self.stitch(xe, y)


def image_to_dst(image_path, dst_path):
    """
    AI इमेज के डिज़ाइन वाले हिस्से को ही भरता है (खाली जगह नहीं),
    असली साइज़ में, 0.4 mm घनत्व, 3.5 mm तक के टाँके, लॉक-स्टिच और ट्रिम के साथ।
    """
    mask = _design_mask(image_path)
    h, w = mask.shape

    pattern = _EmbPattern()
    pattern.add_thread(_make_thread(0, 0, 0))   # काला धागा
    sew = _Sewer(pattern, w // 2, h // 2)   # डिज़ाइन का केंद्र = मशीन का केंद्र
    # किनारे पर 0.2 mm की छूट के साथ "अंदर" की जाँच (किनारे के पिक्सल पर जोड़ फेल न हो)
    loose = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    # हर अलग आकार की अपनी लाइनें
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    jobs = []
    for i in range(1, n):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < MIN_AREA:
            continue
        cm = (labels[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
        for chain in _component_chains(cm):
            lines = []
            for k, (a, b, yy, idx) in enumerate(chain):
                if k % 2 == 0:
                    lines.append((a + x, b + x, yy + y, idx))   # बाएँ → दाएँ
                else:
                    lines.append((b + x, a + x, yy + y, idx))   # दाएँ → बाएँ
            jobs.append(lines)
    if not jobs:
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला")

    cur = (w // 2, h // 2)
    while jobs:
        # सबसे पास वाला आकार पहले (कम से कम जंप)
        j = min(range(len(jobs)),
                key=lambda k: (jobs[k][0][0] - cur[0]) ** 2 + (jobs[k][0][2] - cur[1]) ** 2)
        lines = jobs.pop(j)

        first = True
        prev_sign = 1
        for (xs, xe, y, idx) in lines:
            sign = 1 if xe >= xs else -1
            near = ((xs - sew.x) ** 2 + (y - sew.y) ** 2) ** 0.5 <= MAX_CONNECT
            if (not first) and near and _inside(loose, sew.x, sew.y, xs, y):
                sew.connect(xs, y)                    # अंदर ही अंदर जुड़ें, धागा नहीं कटेगा
            else:
                if not first:
                    sew.lock(sew.x, sew.y, -prev_sign)  # पिछली लाइन पर लॉक
                    sew.trim()                          # धागा काटें
                sew.jump(xs, y)
                sew.lock(xs, y, sign)                   # नई लाइन की शुरुआत पर लॉक
            sew.sew_line(xs, xe, y, (idx % 3) * (FILL_STITCH_LEN // 3))
            prev_sign = sign
            first = False

        sew.lock(sew.x, sew.y, -prev_sign)
        sew.trim()
        cur = (sew.x, sew.y)

    sew.end()
    pyembroidery.write(pattern, dst_path)   # फाइल के .dst नाम से DST फॉर्मेट अपने आप चुनता है
    return dst_path


def image_to_svg(image_path, svg_path):
    """असली वेक्टर SVG: डिज़ाइन की आउटलाइन, असली साइज़ (mm) में।"""
    mask = _design_mask(image_path)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for c in contours:
        c = cv2.approxPolyDP(c, 1.0, True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2)
        paths.append('M' + ' L'.join(f'{int(x)},{int(y)}' for x, y in pts) + ' Z')
    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{w / 10:.1f}mm" height="{h / 10:.1f}mm" viewBox="0 0 {w} {h}">'
           f'<path fill="#000000" fill-rule="evenodd" d="{" ".join(paths)}"/></svg>')
    with open(svg_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    return svg_path


def _demo_image(prompt):
    img = Image.new('RGB', (512, 512), (0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(0, 512, 20):
        d.ellipse([i, i, i + 60, i + 60], outline=(255, 215, 0), width=3)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ============================================================
# AI इमेज बनाने के तरीके: पहले Replicate → फिर OpenAI → फिर Hugging Face
# (Render Environment में इन नामों से Key डालें)
#   REPLICATE_API_TOKEN, OPENAI_API_KEY, HF_TOKEN
# ============================================================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
HF_TOKEN = (os.getenv('HF_TOKEN') or os.getenv('HF_API_KEY') or os.getenv('HUGGINGFACE_API_KEY')
            or os.getenv('HUGGING_FACE_API_KEY') or os.getenv('HUGGINGFACE_TOKEN') or '').strip()
OPENAI_IMAGE_MODEL = (os.getenv('OPENAI_IMAGE_MODEL') or 'gpt-image-1').strip()
HF_IMAGE_MODEL = (os.getenv('HF_IMAGE_MODEL') or 'black-forest-labs/FLUX.1-schnell').strip()


def _to_png(raw_bytes):
    """कोई भी इमेज (PNG/JPG/WEBP) जाँचकर PNG बाइट्स में बदलता है। इमेज न हो तो एरर।"""
    img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _ai_replicate(prompt):
    output = replicate.run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "num_outputs": 1,
            "negative_prompt": "shading, gradient, photo, 3d, shadow, texture, text, watermark, blurry, thin lines",
            "scheduler": "DPMSolverMultistep"
        }
    )
    img_url = output[0] if isinstance(output, list) else str(output)
    return requests.get(str(img_url), timeout=90).content


def _ai_openai(prompt):
    payload = {'model': OPENAI_IMAGE_MODEL, 'prompt': prompt, 'size': '1024x1024', 'n': 1}
    if OPENAI_IMAGE_MODEL.startswith('dall-e'):
        payload['response_format'] = 'b64_json'     # पुराने DALL·E मॉडल के लिए
    else:
        payload['quality'] = 'medium'               # gpt-image मॉडल हमेशा base64 लौटाते हैं
    r = requests.post('https://api.openai.com/v1/images/generations',
                      headers={'Authorization': f'Bearer {OPENAI_API_KEY}',
                               'Content-Type': 'application/json'},
                      json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:400]}')   # असली वजह लॉग में दिखेगी
    item = r.json()['data'][0]
    if item.get('b64_json'):
        return base64.b64decode(item['b64_json'])
    if item.get('url'):
        return requests.get(item['url'], timeout=90).content
    raise RuntimeError('OpenAI से इमेज नहीं मिली')


def _ai_huggingface(prompt):
    r = requests.post(f'https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}',
                      headers={'Authorization': f'Bearer {HF_TOKEN}'},
                      json={'inputs': prompt}, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}: {r.text[:400]}')
    return r.content


def generate_ai_image(prompt):
    """Replicate → OpenAI → Hugging Face. पहला जो चले उसकी PNG इमेज (bytes) लौटाता है, वरना None।"""
    for name, fn, key in (('Replicate', _ai_replicate, REPLICATE_TOKEN),
                          ('OpenAI', _ai_openai, OPENAI_API_KEY),
                          ('Hugging Face', _ai_huggingface, HF_TOKEN)):
        if not key:
            print(f'⏭️ {name}: Key सेट नहीं है, छोड़ दिया')
            continue
        try:
            print(f'Trying {name}...')
            data = _to_png(fn(prompt))
            print(f'✅ {name} से इमेज बन गई')
            return data
        except Exception as e:
            print(f'❌ {name} Failed: {e}')
    return None


# ============================================================
# कन्वर्टर: कोई भी डिज़ाइन इमेज → मल्टी-कलर मशीन फ़ाइल (मल्टी-हेड मशीन के लिए)
# pyembroidery जो फॉर्मेट लिख सकता है: DST, PES, JEF, EXP, VP3, XXX, U01, PEC
# (HUS और SEW सिर्फ़ पढ़े जा सकते हैं, लिखे नहीं जा सकते)
# ============================================================
CONVERT_FORMATS = ['dst', 'pes', 'jef', 'exp', 'vp3', 'xxx', 'u01', 'pec']
try:
    CONVERT_MAX_MM = max(50.0, min(500.0, float(os.getenv('CONVERT_MAX_MM', '300'))))
except ValueError:
    CONVERT_MAX_MM = 300.0           # सबसे बड़ा डिज़ाइन साइज़ (mm)
try:
    CONVERT_COST = max(0, int(os.getenv('CONVERT_CREDIT_COST', '0')))
except ValueError:
    CONVERT_COST = 0                 # हर कन्वर्ज़न पर कटने वाले क्रेडिट (0 = मुफ़्त)
CONVERT_MAX_STITCHES = 250000
CONVERT_SPM = 700                    # समय के अनुमान के लिए: टाँके प्रति मिनट


class _ConvSewer(_Sewer):
    """मल्टी-कलर के लिए: ऊँची टाँका-सीमा और रंग बदलने की सुविधा।"""

    def stitch(self, x, y):
        X, Y = self._pos(x, y)
        self.p.add_stitch_absolute(pyembroidery.STITCH, X, Y)
        self.x, self.y = x, y
        self.count += 1
        if self.count > CONVERT_MAX_STITCHES:
            raise ValueError("डिज़ाइन बहुत भारी है, साइज़ या रंग कम कीजिए।")

    def color_change(self):
        X, Y = self._pos(self.x, self.y)
        self.p.add_stitch_absolute(pyembroidery.COLOR_CHANGE, X, Y)


def _load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("इमेज पढ़ी नहीं जा सकी। कृपया PNG या JPG फ़ाइल डालें।")
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / (img.max() / 255.0 if img.max() > 0 else 1)).astype(np.uint8)
    alpha = None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        img = img[:, :, :3]
    return img, alpha


def _quantize(img, alpha, n_colors, skip_bg):
    """इमेज को n रंगों में बाँटता है। लौटाता है: (लेबल नक्शा, हर लेबल का BGR रंग)। -1 = सिलना नहीं।"""
    h, w = img.shape[:2]
    k = 900.0 / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    img = cv2.bilateralFilter(img, 7, 40, 7)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    cv2.setRNGSeed(1)
    rng = np.random.default_rng(1)
    sample = lab if len(lab) <= 120000 else lab[rng.choice(len(lab), 120000, replace=False)]
    K = max(2, min(n_colors, len(sample)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5)
    _, _, centers = cv2.kmeans(sample, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)

    labels = np.empty(len(lab), dtype=np.int32)
    for s in range(0, len(lab), 200000):
        chunk = lab[s:s + 200000]
        d = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[s:s + 200000] = d.argmin(axis=1)
    labels = labels.reshape(img.shape[:2])

    # हर रंग का असली औसत BGR (धागे का रंग)
    flat_bgr = img.reshape(-1, 3).astype(np.float64)
    flat_lab = labels.reshape(-1)
    colors = []
    for i in range(K):
        sel = flat_lab == i
        colors.append(tuple(int(v) for v in flat_bgr[sel].mean(axis=0)) if sel.any() else (0, 0, 0))

    if alpha is not None:
        labels[alpha < 128] = -1          # पारदर्शी हिस्सा = बैकग्राउंड

    if skip_bg:
        border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        border = border[border >= 0]
        if len(border):
            bg = int(np.bincount(border, minlength=K).argmax())
            labels[labels == bg] = -1     # जो रंग किनारों पर सबसे ज़्यादा है = कपड़ा/बैकग्राउंड
    return labels, colors


def _clean_layer(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.copyMakeBorder(mask, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)[6:-6, 6:-6]
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            out[lbl == i] = 255
    return out


def _sew_mask(sew, mask):
    """एक रंग की परत को सिलता है (ज़िगज़ैग फिल, लॉक-स्टिच, ट्रिम)। AI वाले DST जैसी ही मज़बूत विधि।"""
    loose = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    jobs = []
    for i in range(1, n):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < MIN_AREA:
            continue
        cm = (labels[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
        for chain in _component_chains(cm):
            lines = []
            for k, (a, b, yy, idx) in enumerate(chain):
                if k % 2 == 0:
                    lines.append((a + x, b + x, yy + y, idx))
                else:
                    lines.append((b + x, a + x, yy + y, idx))
            jobs.append(lines)

    cur = (sew.x, sew.y)
    while jobs:
        j = min(range(len(jobs)),
                key=lambda k: (jobs[k][0][0] - cur[0]) ** 2 + (jobs[k][0][2] - cur[1]) ** 2)
        lines = jobs.pop(j)
        first = True
        prev_sign = 1
        for (xs, xe, y, idx) in lines:
            sign = 1 if xe >= xs else -1
            near = ((xs - sew.x) ** 2 + (y - sew.y) ** 2) ** 0.5 <= MAX_CONNECT
            if (not first) and near and _inside(loose, sew.x, sew.y, xs, y):
                sew.connect(xs, y)
            else:
                if not first:
                    sew.lock(sew.x, sew.y, -prev_sign)
                    sew.trim()
                sew.jump(xs, y)
                sew.lock(xs, y, sign)
            sew.sew_line(xs, xe, y, (idx % 3) * (FILL_STITCH_LEN // 3))
            prev_sign = sign
            first = False
        sew.lock(sew.x, sew.y, -prev_sign)
        sew.trim()
        cur = (sew.x, sew.y)


def convert_image_to_machine(src_path, out_base, formats, size_mm, n_colors, skip_bg):
    """
    इमेज → मल्टी-कलर मशीन फ़ाइलें।
    out_base: बिना एक्सटेंशन का पाथ (जैसे /x/conv_1_ab12). लौटाता है: dict (files, stats, colors, preview)
    """
    img, alpha = _load_bgr(src_path)
    labels, colors = _quantize(img, alpha, n_colors, skip_bg)

    fg = labels >= 0
    if not fg.any():
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला। 'बैकग्राउंड न सिलें' बंद करके देखें।")
    ys, xs = np.where(fg)
    labels = labels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = labels.shape
    scale = (size_mm * 10.0) / max(ch, cw)
    W, H = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    ppm = cw / (W / 10.0)            # छोटी इमेज के पिक्सल प्रति mm

    # हर रंग की परत की जानकारी और सिलाई का क्रम
    layers = []
    for k in range(len(colors)):
        small = (labels == k).astype(np.uint8) * 255
        area = int((small > 0).sum())
        if area == 0:
            continue
        dist = cv2.distanceTransform(small, cv2.DIST_L2, 3)
        thick_mm = 2.0 * float(dist[small > 0].mean()) / ppm
        b, g, r = colors[k]
        lum = 0.114 * b + 0.587 * g + 0.299 * r
        thin = thick_mm < 2.5        # आउटलाइन/बारीक डिटेल = सबसे ऊपर, आख़िर में
        key = (1, -lum) if thin else (0, -area)   # पहले बड़े हिस्से, फिर बारीक (हल्के से गहरा)
        layers.append((key, k, area, thin))
    layers.sort(key=lambda t: t[0])
    if not layers:
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला।")

    pattern = _EmbPattern()
    sew = _ConvSewer(pattern, W // 2, H // 2)
    sigma = max(1.0, scale * 0.7)
    used = []
    for (_, k, area, thin) in layers:
        m = (labels == k).astype(np.uint8) * 255
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), sigma)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        if not thin:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))  # ऊपर की परत से 0.3 mm दबाव, गैप नहीं
        m = _clean_layer(m)
        if not m.any():
            continue

        b, g, r = colors[k]
        pattern.add_thread(_make_thread(r, g, b))
        if used:
            sew.color_change()
        before = sew.count
        _sew_mask(sew, m)
        used.append({'k': k, 'bgr': colors[k], 'stitches': sew.count - before})
    if not used:
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला।")
    sew.end()

    files = {}
    for fmt in formats:
        path = f'{out_base}.{fmt}'
        pyembroidery.write(pattern, path)
        files[fmt] = path

    # रंग प्रीव्यू (सिलाई के क्रम में)
    prev = np.full((ch, cw, 3), 235, np.uint8)
    for u in used:
        prev[labels == u['k']] = u['bgr']
    pk = 700.0 / max(ch, cw)
    if pk < 1:
        prev = cv2.resize(prev, (int(cw * pk), int(ch * pk)), interpolation=cv2.INTER_AREA)
    preview_path = f'{out_base}_p.png'
    cv2.imwrite(preview_path, prev)

    total = sew.count
    colors_out = []
    for i, u in enumerate(used):
        b, g, r = u['bgr']
        colors_out.append({'needle': i + 1, 'hex': '#%02x%02x%02x' % (r, g, b), 'stitches': u['stitches']})
    stats = {
        'width_mm': round(W / 10.0, 1), 'height_mm': round(H / 10.0, 1),
        'stitches': total, 'colors': len(used), 'color_changes': len(used) - 1,
        'minutes': round(total / float(CONVERT_SPM), 1),
        'density_mm': FILL_SPACING / 10.0, 'stitch_type': 'फिल (ज़िगज़ैग)',
    }
    return {'files': files, 'stats': stats, 'colors': colors_out, 'preview': preview_path}


def _threads_txt(stats, colors):
    lines = ['DESIGN SIZE: %.1f x %.1f mm' % (stats['width_mm'], stats['height_mm']),
             'STITCHES: %d   COLORS: %d   COLOR CHANGES: %d' % (stats['stitches'], stats['colors'], stats['color_changes']),
             'STITCH TYPE: FILL   DENSITY: %.1f mm' % stats['density_mm'],
             '']
    for c in colors:
        lines.append('NEEDLE %d : %s : %d stitches' % (c['needle'], c['hex'].upper(), c['stitches']))
    return '\r\n'.join(lines) + '\r\n'


# ============================================================
# ROUTES: कन्वर्टर (इमेज अपलोड → मल्टी-कलर मशीन फ़ाइल)
# ============================================================
@app.route('/converter')
@login_required
def converter():
    return render_template('converter.html', user=current_user,
                           max_mm=int(CONVERT_MAX_MM), cost=CONVERT_COST)


@app.route('/api/convert', methods=['POST'])
@login_required
def api_convert():
    if not current_user.is_trial_active():
        return jsonify({'status': 'error', 'message': 'आपका 15 दिन का ट्रायल खत्म हो गया है! कृपया प्लान खरीदें।'}), 402
    if CONVERT_COST and current_user.credits < CONVERT_COST:
        return jsonify({'status': 'error', 'message': 'क्रेडिट्स खत्म हो गए हैं! कृपया प्लान खरीदें।'}), 402

    f = request.files.get('image')
    if not f or not f.filename:
        return jsonify({'status': 'error', 'message': 'कृपया पहले डिज़ाइन की इमेज चुनें।'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
        return jsonify({'status': 'error', 'message': 'कृपया PNG या JPG इमेज डालें।'}), 400

    fmt = (request.form.get('format') or 'dst').lower()
    if fmt != 'all' and fmt not in CONVERT_FORMATS:
        fmt = 'dst'
    try:
        size_mm = float(request.form.get('size_mm') or 100)
    except ValueError:
        size_mm = 100.0
    size_mm = max(20.0, min(CONVERT_MAX_MM, size_mm))
    try:
        n_colors = int(request.form.get('colors') or 6)
    except ValueError:
        n_colors = 6
    n_colors = max(2, min(12, n_colors))
    skip_bg = (request.form.get('skip_bg', '1') != '0')

    base = f'conv_{current_user.id}_{uuid.uuid4().hex[:10]}'
    out_base = os.path.join(app.config['UPLOAD_FOLDER'], base)
    src_path = out_base + '_src' + ext
    f.save(src_path)
    try:
        formats = CONVERT_FORMATS if fmt == 'all' else [fmt]
        res = convert_image_to_machine(src_path, out_base, formats, size_mm, n_colors, skip_bg)
        if fmt == 'all':
            with zipfile.ZipFile(out_base + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
                for ff, pth in res['files'].items():
                    z.write(pth, f'DESIGN.{ff.upper()}')
                z.writestr('THREADS.TXT', _threads_txt(res['stats'], res['colors']))
            download_url = f'/converted/{base}.zip'
        else:
            download_url = f'/converted/{base}.{fmt}'
        if CONVERT_COST:
            current_user.credits -= CONVERT_COST
            db.session.commit()
        return jsonify({
            'status': 'success',
            'download_url': download_url,
            'format': fmt,
            'preview_url': f'/converted/{base}_p.png',
            'stats': res['stats'],
            'colors': res['colors'],
            'remaining_credits': current_user.credits,
        })
    except Exception as e:
        print('Convert error:', e)
        traceback.print_exc()
        where = ''
        if current_user.is_admin or os.getenv('SHOW_ERROR_DETAILS') == '1':   # गड़बड़ की जगह सिर्फ़ admin को (या जाँच के लिए)
            try:
                fr = traceback.extract_tb(e.__traceback__)[-1]
                where = f' [{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}]'
            except Exception:
                pass
        return jsonify({'status': 'error', 'message': f'कन्वर्ट नहीं हो पाया: {e}{where}'}), 400
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass


@app.route('/converted/<name>')
@login_required
def converted_file(name):
    # सिर्फ़ अपने ही बनाए फ़ाइल नाम (conv_<user id>_<10 hex>...) डाउनलोड हो सकते हैं
    if not re.fullmatch(r'conv_%d_[0-9a-f]{10}(_p)?\.[a-z0-9]{2,4}' % current_user.id, name):
        abort(404)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    if not os.path.exists(path):
        abort(404)
    if name.endswith('_p.png'):
        return send_file(path, mimetype='image/png')
    ext = name.rsplit('.', 1)[-1]
    download_name = 'DESIGNS.zip' if ext == 'zip' else 'DESIGN.' + ext.upper()
    return send_file(path, as_attachment=True, download_name=download_name)


# ============================================================
# ROUTES: AUTH
# ============================================================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        user = User.query.filter_by(username=u).first()
        if user and user.check_password(p):
            login_user(user, remember=True)
            flash(f'स्वागत है {user.username}! 🎉', 'success')
            return redirect(url_for('home'))
        flash('गलत यूज़रनेम या पासवर्ड!', 'error')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        e = request.form.get('email', '').strip()
        p = request.form.get('password', '')
        if not (u and e and p):
            flash('सभी फील्ड भरें!', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(username=u).first():
            flash('यह यूज़रनेम पहले से मौजूद है!', 'error')
            return redirect(url_for('register'))
        if User.query.filter_by(email=e).first():
            flash('यह ईमेल पहले से रजिस्टर है!', 'error')
            return redirect(url_for('register'))
        
        # नया यूज़र: 20 क्रेडिट और 15 दिन का ट्रायल
        user = User(username=u, email=e, credits=20, plan='trial')
        user.set_password(p)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash('रजिस्ट्रेशन सफल! 15 दिन का फ्री ट्रायल और 20 क्रेडिट्स मिले 🎁', 'success')
        return redirect(url_for('home'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('लॉगआउट हो गए।', 'info')
    return redirect(url_for('login'))


# ============================================================
# ROUTES: PAGES
# ============================================================
@app.route('/')
@login_required
def home():
    return render_template('home.html', trial_active=current_user.is_trial_active())


@app.route('/generator')
@login_required
def generator():
    return render_template('generator.html', user=current_user)


@app.route('/pricing')
@login_required
def pricing():
    return render_template('pricing.html', plans=PLANS, user=current_user)


@app.route('/dashboard')
@login_required
def dashboard():
    designs = Design.query.filter_by(user_id=current_user.id).order_by(Design.created_at.desc()).all()
    return render_template('dashboard.html', designs=designs, user=current_user)


# ============================================================
# ROUTES: DOWNLOAD
# ============================================================
@app.route('/download/<int:design_id>/<filetype>')
@login_required
def download_design(design_id, filetype):
    d = Design.query.get_or_404(design_id)
    if d.user_id != current_user.id:
        abort(403)
    if filetype == 'image' and d.image_file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], d.image_file)
    elif filetype == 'dst' and d.dst_file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], d.dst_file)
    elif filetype == 'svg' and d.svg_file:
        path = os.path.join(app.config['UPLOAD_FOLDER'], d.svg_file)
    else:
        abort(404)
    if not os.path.exists(path):
        abort(404)
    names = {'image': f'design_{d.id}.png', 'dst': f'D{d.id}.DST', 'svg': f'design_{d.id}.svg'}
    return send_file(path, as_attachment=True, download_name=names[filetype])


# ============================================================
# ROUTES: API (AI GENERATION & OTHERS)
# ============================================================
@app.route('/api/generate', methods=['POST'])
@login_required
def api_generate():
    # ट्रायल और क्रेडिट चेक
    if not current_user.is_trial_active():
        return jsonify({'status': 'error', 'message': 'आपका 15 दिन का ट्रायल खत्म हो गया है! कृपया प्लान खरीदें।'}), 402
    if current_user.credits <= 0:
        return jsonify({'status': 'error', 'message': 'क्रेडिट्स खत्म हो गए हैं! कृपया प्लान खरीदें।'}), 402

    data = request.get_json() or {}
    prompt = (data.get('prompt') or '').strip()
    if not prompt:
        return jsonify({'status': 'error', 'message': 'कृपया पहले प्रॉम्प्ट लिखें!'}), 400

    enhanced = (
        f"{prompt}, flat 2D embroidery design, bold solid shapes, "
        f"black design on plain white background, high contrast, "
        f"no shading, no gradient, no texture, no text, "
        f"thick clean outlines, simple, centered, vector style"
    )

    try:
        # ---------- AI GENERATION (Replicate → OpenAI → Hugging Face) ----------
        if REPLICATE_TOKEN or OPENAI_API_KEY or HF_TOKEN:
            img_bytes = generate_ai_image(enhanced)
            if img_bytes is None:
                return jsonify({'status': 'error',
                                'message': 'अभी कोई भी AI सेवा काम नहीं कर रही (क्रेडिट/सेटिंग की दिक्कत)। '
                                           'कृपया थोड़ी देर बाद कोशिश करें। आपका क्रेडिट नहीं कटा।'}), 503
        else:
            img_bytes = _demo_image(enhanced)

        # ---------- SAVE IMAGE ----------
        fname = f"design_{uuid.uuid4().hex[:10]}.png"
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        with open(img_path, 'wb') as f:
            f.write(img_bytes)

        # ---------- CONVERT TO DST (Fully Filled) ----------
        dst_name = fname.replace('.png', '.dst')
        dst_path = os.path.join(app.config['UPLOAD_FOLDER'], dst_name)
        try:
            image_to_dst(img_path, dst_path)
        except Exception as e:
            print('DST conversion failed:', e)
            traceback.print_exc()
            dst_name = None

        # ---------- SVG (असली वेक्टर) ----------
        svg_name = fname.replace('.png', '.svg')
        svg_path = os.path.join(app.config['UPLOAD_FOLDER'], svg_name)
        try:
            image_to_svg(img_path, svg_path)
        except Exception as e:
            print('SVG conversion failed:', e)
            traceback.print_exc()
            svg_name = None

        # ---------- SAVE TO DB ----------
        design = Design(
            user_id=current_user.id,
            prompt=prompt,
            image_file=fname,
            dst_file=dst_name,
            svg_file=svg_name
        )
        current_user.credits -= 1
        db.session.add(design)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'design_id': design.id,
            'image_url': f'/static/generated/{fname}',
            'dst_ready': bool(dst_name),
            'svg_url': (f'/static/generated/{svg_name}' if svg_name else None),
            'remaining_credits': current_user.credits,
            'message': ('डिज़ाइन सफलतापूर्वक तैयार! 🎉' if dst_name else
                        'इमेज बन गई, पर इसमें से मशीन फ़ाइल नहीं बन पाई। कृपया सरल प्रॉम्प्ट आज़माएँ।')
        })

    except Exception as e:
        print('Generation error:', e)
        return jsonify({'status': 'error', 'message': f'एरर: {str(e)}'}), 500


@app.route('/api/check-credits')
@login_required
def api_check_credits():
    return jsonify({
        'status': 'success',
        'credits': current_user.credits,
        'plan': current_user.plan,
        'trial_active': current_user.is_trial_active()
    })


@app.route('/api/submit-payment', methods=['POST'])
@login_required
def api_submit_payment():
    data = request.get_json() or {}
    plan_key = data.get('plan', '').lower()
    utr = data.get('utr', '').strip()
    
    if plan_key not in PLANS or not utr:
        return jsonify({'status': 'error', 'message': 'कृपया सही जानकारी भरें।'}), 400

    new_pay = PaymentRequest(user_id=current_user.id, plan_key=plan_key, amount=PLANS[plan_key]['price'], utr_number=utr)
    db.session.add(new_pay)
    db.session.commit()

    return jsonify({'status': 'success', 'message': 'पेमेंट रिक्वेस्ट भेज दी गई है!'})


# ============================================================
# ROUTES: ADMIN
# ============================================================
@app.route('/admin/payments')
@login_required
def admin_payments():
    if not current_user.is_admin:
        abort(403)
    payments = PaymentRequest.query.order_by(PaymentRequest.created_at.desc()).all()
    return render_template('admin_payments.html', payments=payments, plans=PLANS)


@app.route('/admin/approve/<int:pay_id>')
@login_required
def admin_approve(pay_id):
    if not current_user.is_admin:
        abort(403)
    pay = PaymentRequest.query.get_or_404(pay_id)
    if pay.status == 'pending':
        pay.status = 'approved'
        user = User.query.get(pay.user_id)
        plan = PLANS[pay.plan_key]
        
        # यहाँ यूज़र का अकाउंट एक्टिवेट होगा
        user.plan = pay.plan_key
        user.credits += plan['credits']
        user.is_paid_user = True
        db.session.commit()
        flash(f'{user.username} का {plan["name"]} प्लान एक्टिवेट हो गया!', 'success')
    return redirect(url_for('admin_payments'))

# ============================================================
# BOOTSTRAP DB + ADMIN (Render Deployment के लिए 100% सही)
# ============================================================
with app.app_context():
    try:
        # यह लाइन डेटाबेस से कनेक्शन को टेस्ट करती है
        db.engine.connect() 
        db.create_all()
        print("✅ Tables created successfully on PostgreSQL!")
        
        ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', '').strip()
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', email='admin@surat.local', is_admin=True)
            admin.plan = 'business'
            admin.credits = 9999
            admin.is_paid_user = True
            admin.set_password(ADMIN_PASSWORD or secrets.token_urlsafe(24))
            db.session.add(admin)
            db.session.commit()
            print('✅ Admin अकाउंट बन गया')
        elif ADMIN_PASSWORD:
            admin.set_password(ADMIN_PASSWORD)   # Render Environment वाला पासवर्ड ही चलेगा
            db.session.commit()
        elif admin.check_password('admin123'):
            admin.set_password(secrets.token_urlsafe(24))   # पुराना कमज़ोर पासवर्ड बंद
            db.session.commit()
            print('⚠️ पुराना admin123 पासवर्ड बंद कर दिया गया')
        if not ADMIN_PASSWORD:
            print('⚠️ Render Environment में ADMIN_PASSWORD डालिए, वरना admin लॉगिन नहीं होगा')
    except Exception as e:
        print(f'❌ Database initialization error: {e}')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
