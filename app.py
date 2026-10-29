import os
import io
import uuid
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
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'), static_folder=os.path.join(base_dir, 'static'))
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'surat-embroidery-secret-key-change-me')

# डेटाबेस URL सेट करना (Render PostgreSQL के लिए)
db_url = os.getenv('DATABASE_URL', 'sqlite:///./database.db')

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
# HELPER: AI IMAGE → FULLY FILLED MACHINE .DST CONVERTER
# ============================================================
def image_to_dst(image_path, dst_path):
    """
    यह फंक्शन इमेज के हर काले हिस्से (डिज़ाइन) को ज़िगज़ैग (Satin Stitch) 
    से पूरा भरता है, जिससे मशीन पूरा फूल/पत्ती/बॉर्डर बनाती है।
    """
    # 1. इमेज लोड करें (ब्लैक एंड व्हाइट)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("इमेज लोड नहीं हो पाई!")
    
    # मशीन के हूप के हिसाब से 500x500 में रीसाइज करें
    img = cv2.resize(img, (500, 500))
    
    # 2. इमेज को ब्लैक एंड व्हाइट (थ्रेशोल्ड) करें
    # जो डिज़ाइन है वो सफ़ेद (255) हो जाएगा और बैकग्राउंड काला (0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. नॉइज़ (छोटे-छोटे धब्बे) साफ़ करें ताकि सुई बार-बार न टूटे
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. पैटर्न शुरू करें
    pattern = pyembroidery.EmbPattern()
    pattern.add_thread(pyembroidery.EmbThread(color=0x000000, name='Color 1'))

    # 5. ज़िगज़ैग (Satin) फिलिंग का लॉजिक
    # हम हर 3 पिक्सल की दूरी पर ऊपर से नीचे तक स्कैन करेंगे
    stitch_spacing = 3 
    
    for y in range(0, 500, stitch_spacing):
        row = thresh[y]
        dark_pixels = np.where(row > 0)[0]
        
        if len(dark_pixels) > 0:
            # डिज़ाइन की शुरुआत और अंत खोजें
            start_x = dark_pixels[0]
            end_x = dark_pixels[-1]
            
            # अगर डिज़ाइन की चौड़ाई 3 पिक्सल से ज़्यादा है, तो उसे भरें
            if end_x - start_x > 3:
                # ज़िगज़ैग (Satin Stitch) बनाने के लिए आगे-पीछे टांके लगाते हैं
                pattern.add_stitch_absolute(pyembroidery.STITCH, start_x, y) # बाएं से दाएं
                pattern.add_stitch_absolute(pyembroidery.STITCH, end_x, y)   # दाएं से बाएं
                
                # अगली लाइन पर जाने के लिए सुई उठाएं (Jump)
                pattern.add_stitch_absolute(pyembroidery.JUMP, start_x, y + stitch_spacing)

    # 6. डिज़ाइन खत्म करें और .DST फाइल सेव करें
    pattern.end()
    pyembroidery.write_dst(pattern, dst_path)
    
    return dst_path


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
    return send_file(path, as_attachment=True)


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
        f"{prompt}, high quality embroidery vector design, seamless floral "
        f"pattern, zari work, intricate stitches, black background, "
        f"clean sharp lines, punch-card style, textile pattern"
    )

    try:
        # ---------- AI GENERATION ----------
        if REPLICATE_TOKEN:
            output = replicate.run(
                "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
                input={
                    "prompt": enhanced,
                    "width": 1024,
                    "height": 1024,
                    "num_outputs": 1,
                    "scheduler": "DPMSolverMultistep"
                }
            )
            img_url = output[0] if isinstance(output, list) else str(output)
            img_bytes = requests.get(img_url, timeout=90).content
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
            dst_name = None

        # ---------- SVG ----------
        svg_name = fname.replace('.png', '.svg')
        svg_path = os.path.join(app.config['UPLOAD_FOLDER'], svg_name)
        with open(svg_path, 'w') as f:
            f.write(f'<?xml version="1.0"?>\n'
                    f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'xmlns:xlink="http://www.w3.org/1999/xlink" '
                    f'width="1024" height="1024">'
                    f'<image width="1024" height="1024" '
                    f'xlink:href="/static/generated/{fname}"/></svg>')

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
            'svg_url': f'/static/generated/{svg_name}',
            'remaining_credits': current_user.credits,
            'message': 'डिज़ाइन सफलतापूर्वक तैयार! 🎉'
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
        
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@surat.local', is_admin=True)
            admin.set_password('admin123')
            admin.plan = 'business'
            admin.credits = 9999
            admin.is_paid_user = True
            db.session.add(admin)
            db.session.commit()
            print('✅ Default admin created: admin / admin123')
    except Exception as e:
        print(f'❌ Database initialization error: {e}')
      import traceback

@app.errorhandler(500)
def show_error(e):
    return f"<pre>{traceback.format_exc()}</pre>", 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
