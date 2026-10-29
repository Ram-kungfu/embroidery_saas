import os
import io
import uuid
import secrets
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
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("इमेज लोड नहीं हो पाई!")

    h, w = img.shape
    k = 1024.0 / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 1. अपने आप सही थ्रेशोल्ड (Otsu)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. बैकग्राउंड पहचानें (अगर बैकग्राउंड सफ़ेद है, तो उल्टा करें)
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # 3. अगर डिज़ाइन सिर्फ एक पतली लाइन (Outline) है, तो उसे मोटा करें ताकि सिलाई हो सके
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel_thick, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("इमेज में सिलने लायक डिज़ाइन नहीं मिला")

    # 4. क्रॉपिंग और असली साइज़ (mm) में बदलना
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = mask.shape
    s = (DESIGN_SIZE_MM * 10.0) / max(ch, cw)
    mask = cv2.resize(mask, (max(1, int(round(cw * s))), max(1, int(round(ch * s)))),
                      interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 5. बहुत छोटे धब्बों को हटाना (नॉइज़ क्लीनिंग)
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
    AI इमेज के डिज़ाइन वाले हिस्से को ही भरता है (Outline हो या Solid),
    असली साइज़ में, 0.4 mm घनत्व के साथ।
    """
    try:
        mask = _design_mask(image_path)
    except Exception as e:
        raise ValueError(f"Masking failed: {e}")
        
    h, w = mask.shape

    pattern = pyembroidery.EmbPattern()
    pattern.add_thread(pyembroidery.EmbThread(color=0x000000, name='Color 1'))
    sew = _Sewer(pattern, w // 2, h // 2)
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
            
    if not jobs:
        raise ValueError("इमेज में सिलने लायक कोई लाइन नहीं मिली")

    cur = (w // 2, h // 2)
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
                
            # यहाँ टांके लगाएं
            sew.sew_line(xs, xe, y, (idx % 3) * (FILL_STITCH_LEN // 3))
            prev_sign = sign
            first = False

        sew.lock(sew.x, sew.y, -prev_sign)
        sew.trim()
        cur = (sew.x, sew.y)

    pattern.end()
    pyembroidery.write_dst(pattern, dst_path)
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

    # ⚡ AI Prompt सुधार: हर बार नया डिज़ाइन बनेगा
    enhanced = (
        f"{prompt}, highly detailed unique intricate {uuid.uuid4().hex[:4]}, "
        f"solid filled embroidery vector design, thick bold shapes, "
        f"plain pure white background, no background elements, "
        f"no shading, no gradients, no texture, no text, "
        f"clean thick edges, centered, modern, creative"
    )

    try:
        # ============================================================
        # स्मार्ट AI फ़ॉलबैक (Replicate -> OpenAI -> Hugging Face)
        # ============================================================
        img_bytes = None
        error_logs = []

        # 1. पहले Replicate ट्राई करें
        if REPLICATE_TOKEN:
            try:
                print("Trying Replicate AI...")
                output = replicate.run(
                    "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                    input={
                        "prompt": enhanced,
                        "width": 1024,
                        "height": 1024,
                        "num_outputs": 1,
                        "negative_prompt": "shading, gradient, photo, 3d, shadow, texture, text, watermark, blurry, thin lines, black background",
                        "scheduler": "DPMSolverMultistep"
                    }
                )
                img_url = output[0] if isinstance(output, list) else str(output)
                img_bytes = requests.get(img_url, timeout=90).content
                print("✅ Replicate Success")
            except Exception as e:
                print(f"❌ Replicate Failed: {e}")
                error_logs.append(f"Replicate: {e}")

        # 2. अगर Replicate फेल हो, तो OpenAI (ChatGPT) ट्राई करें
        if not img_bytes:
            OPENAI_KEY = os.getenv('OPENAI_API_KEY', '')
            if OPENAI_KEY:
                try:
                    print("Trying OpenAI DALL-E...")
                    headers = {
                        "Authorization": f"Bearer {OPENAI_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "dall-e-3",
                        "prompt": enhanced,
                        "n": 1,
                        "size": "1024x1024",
                        "response_format": "url"
                    }
                    res = requests.post("https://api.openai.com/v1/images/generations", 
                                        headers=headers, json=payload, timeout=90)
                    res.raise_for_status()
                    img_url = res.json()['data'][0]['url']
                    img_bytes = requests.get(img_url, timeout=90).content
                    print("✅ OpenAI Success")
                except Exception as e:
                    print(f"❌ OpenAI Failed: {e}")
                    error_logs.append(f"OpenAI: {e}")

        # 3. अगर दोनों फेल हो, तो Hugging Face ट्राई करें
        if not img_bytes:
            HF_TOKEN = os.getenv('HUGGINGFACE_API_KEY', '')
            if HF_TOKEN:
                try:
                    print("Trying Hugging Face...")
                    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                    payload = {"inputs": enhanced}
                    res = requests.post("https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell", 
                                        headers=headers, json=payload, timeout=90)
                    res.raise_for_status()
                    img_bytes = res.content
                    print("✅ Hugging Face Success")
                except Exception as e:
                    print(f"❌ Hugging Face Failed: {e}")
                    error_logs.append(f"HuggingFace: {e}")

        # 4. अगर तीनों फेल हो जाएं, तो डेमो इमेज बनाएं
        if not img_bytes:
            print("⚠️ All AI failed. Using Demo Image.")
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
            dst_name = None

        # ---------- SVG (असली वेक्टर) ----------
        svg_name = fname.replace('.png', '.svg')
        svg_path = os.path.join(app.config['UPLOAD_FOLDER'], svg_name)
        try:
            image_to_svg(img_path, svg_path)
        except Exception as e:
            print('SVG conversion failed:', e)
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
