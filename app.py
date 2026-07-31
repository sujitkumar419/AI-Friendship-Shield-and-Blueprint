import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="AI Friend Shield & Blueprint", page_icon="🤝", layout="wide")

# ------------------------------------------------------------
# Custom CSS -> Premium Dark-Blue Cyber Theme + Neon Glow Cards
# ------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #050814; color: #f1f5f9; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px; background-color: #0f172a; padding: 12px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; font-size: 16px; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }

    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 26px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        backdrop-filter: blur(12px);
        margin-bottom: 22px;
    }

    .status-true {
        color: #10b981; font-size: 26px; font-weight: bold;
        text-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
    }
    .status-fake {
        color: #f43f5e; font-size: 26px; font-weight: bold;
        text-shadow: 0 0 12px rgba(244, 63, 94, 0.6);
    }

    .glow-box-green {
        background: rgba(16, 185, 129, 0.1); border-left: 6px solid #10b981;
        border-radius: 8px; padding: 20px; margin-top: 15px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
    }
    .glow-box-red {
        background: rgba(244, 63, 94, 0.1); border-left: 6px solid #f43f5e;
        border-radius: 8px; padding: 20px; margin-top: 15px;
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.1);
    }
    .glow-box-purple {
        background: rgba(168, 85, 247, 0.1); border-left: 6px solid #a855f7;
        border-radius: 8px; padding: 20px; margin-top: 15px;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.1);
    }

    .suggestion-header { font-size: 20px; font-weight: bold; margin-bottom: 8px; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. LOAD ML ASSETS (.pkl files from your Jupyter notebook)
# ============================================================
@st.cache_resource
def load_ml_assets():
    with open('knn_friend_blueprint_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('knn_friend_blueprint_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('knn_model_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features


try:
    knn_model, scaler, model_features = load_ml_assets()
except FileNotFoundError:
    st.error("⚠️ Error: Please copy your 3 .pkl files from the Jupyter folder into this app's folder first!")
    st.stop()

# ============================================================
# 3. DUAL LANGUAGE TRANSLATION DICTIONARY
# ============================================================
translations = {
    "English": {
        "title": "🤝 AI FRIENDSHIP SHIELD & BLUEPRINT",
        "subtitle": "Trained KNN Model Engine (89.10% Accuracy) — Evaluate Traits & Get Blueprint Solutions",
        "lang_select": "Choose Interface Language / भाषा चुनें",
        "tab1": "🔍 Evaluate Present Friend",
        "tab2": "🌟 My AI Perfect Friend Blueprint & 2D Space",
        "header_user": "Step 1: Your Personal Profile",
        "header_friend": "Step 2: Friend Behavioral Inputs",
        "profession": "Select Your Profession",
        "weakness": "Select Your Biggest Weakness/Insecurity",
        "f_help": "Helps In Need Score (1 = Selfish, 10 = Always Stands By You)",
        "f_secrets": "Keeps Your Personal Secrets Safe (%)",
        "f_backbite": "Backbiting/Gossip Frequency Count (Behind your back)",
        "f_calls": "Calls/Texts ONLY when they need some help?",
        "f_toxic_circle": "How many of their other close friends are toxic/manipulative (0-3)?",
        "f_respect": "Do they respect parents, waiters, and others generally?",
        "btn_scan": "⚡ PROCESS ASALI ML TRUST AUDIT",
        "res_true": "✅ VERIFIED STATUS: TRUE & TRUSTWORTHY FRIEND",
        "res_fake": "🚨 WARNING STATUS: HIGH RISK / FAKE MATLABI FRIEND",
        "bp_title": "🧭 Your AI-Generated Perfect Friend Blueprint",
        "plot_title": "2D Friendship Trust Space (Your Friend vs Population Baselines)",
        "no_data": "Please run the Trust Audit under Tab 1 first to compile your personalized blueprint.",
        "shield_plan": "🛡️ Shield: Emergency Protective Action Plan For You",
        "ideal_match": "🎯 Target: The Perfect Friend Profile You Actually Need",
        "red_flags": "🚫 Ban: Critical Red Flags To Avoid Next Time",
        "yes": "Yes",
        "no": "No",
    },
    "Hindi": {
        "title": "🤝 AI फ्रेंडशिप शील्ड और ब्लूप्रिंट",
        "subtitle": "ट्रेन्ड KNN मॉडल इंजन (89.10% सटीकता) — आदतें परखें और ब्लूप्रिंट समाधान पाएं",
        "lang_select": "Choose Interface Language / भाषा चुनें",
        "tab1": "🔍 वर्तमान दोस्त की जांच",
        "tab2": "🌟 मेरा AI परफेक्ट फ्रेंड ब्लूप्रिंट और 2D ग्राफ़",
        "header_user": "चरण 1: आपकी अपनी प्रोफ़ाइल",
        "header_friend": "चरण 2: दोस्त की आदतें और व्यवहार दर्ज करें",
        "profession": "अपना पेशा/काम चुनें",
        "weakness": "अपनी सबसे बड़ी कमजोरी/चिंता चुनें",
        "f_help": "मुश्किल में मदद करने का स्कोर (1 = स्वार्थी, 10 = हमेशा साथ खड़ा रहने वाला)",
        "f_secrets": "आपकी व्यक्तिगत गुप्त बातें (Secrets) सुरक्षित रखने की दर (%)",
        "f_backbite": "आपके पीठ पीछे चुगली/गॉसिप करने की संख्या",
        "f_calls": "क्या वे केवल तभी कॉल/मेसेज करते हैं जब उन्हें कोई काम होता है?",
        "f_toxic_circle": "उनके बाकी दोस्तों में से कितने मतलबी/टॉक्सिक हैं (0-3)?",
        "f_respect": "क्या वे माता-पिता, वेटर्स और अन्य लोगों का सम्मान करते हैं?",
        "btn_scan": "⚡ असली ML ट्रस्ट ऑडिट चलाएं",
        "res_true": "✅ सत्यापित स्थिति: सच्चा और विश्वासपात्र मित्र",
        "res_fake": "🚨 चेतावनी स्थिति: हाई रिस्क / मतलबी और नकली दोस्त",
        "bp_title": "🧭 आपका AI-जनरेटेड परफेक्ट फ्रेंड ब्लूप्रिंट",
        "plot_title": "2D फ्रेंडशिप ट्रस्ट स्पेस (आपका दोस्त बनाम बाकी लोगों का डेटा)",
        "no_data": "अपनी व्यक्तिगत ब्लूप्रिंट रिपोर्ट तैयार करने के लिए कृपया पहले टैब 1 में ट्रस्ट ऑडिट चलाएं।",
        "shield_plan": "🛡️ Shield: आपके लिए तत्काल सुरक्षा और व्यावहारिक उपाय",
        "ideal_match": "🎯 Target: वह परफेक्ट दोस्त जो वास्तव में आपके जीवन में होना चाहिए",
        "red_flags": "🚫 Ban: अगली बार के लिए महत्वपूर्ण रेड फ्लैग्स जिनसे बचना है",
        "yes": "हाँ",
        "no": "नहीं",
    }
}

selected_lang = st.sidebar.selectbox(translations["English"]["lang_select"], ["English", "Hindi"])
text = translations[selected_lang]

# ============================================================
# 4. TITLE LAYOUT
# ============================================================
st.title(text["title"])
st.write(text["subtitle"])
st.markdown("---")

tab1, tab2 = st.tabs([text["tab1"], text["tab2"]])

# ============================================================
# 5. SESSION STATE (persists prediction across tabs)
# ============================================================
if 'audit_complete' not in st.session_state:
    st.session_state.audit_complete = False
if 'user_weak' not in st.session_state:
    st.session_state.user_weak = ""
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = 0
if 'current_coords' not in st.session_state:
    st.session_state.current_coords = None

# ============================================================
# TAB 1 — EVALUATE PRESENT FRIEND
# ============================================================
with tab1:
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown(f"<div class='glass-card'><h4>{text['header_user']}</h4></div>", unsafe_allow_html=True)
        user_prof = st.selectbox(
            text["profession"],
            ['Student', 'Software Engineer', 'Business Owner', 'Job Seeker']
        )

    with col_u2:
        st.markdown("<div class='glass-card'><h4>&nbsp;</h4></div>", unsafe_allow_html=True)
        user_weak = st.selectbox(
            text["weakness"],
            ['Study/Work Focus', 'Emotional/Trusts Easily', 'Money Management', 'Anxiety/Stress']
        )

    st.markdown(f"<div class='glass-card'><h4>{text['header_friend']}</h4></div>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st_help = st.slider(text["f_help"], 1.0, 10.0, value=5.0, step=0.5)
        st_secrets = st.slider(text["f_secrets"], 10, 100, value=60)
        st_backbite = st.number_input(text["f_backbite"], min_value=0, max_value=20, value=3)

    with col_f2:
        calls_choice = st.radio(text["f_calls"], [text["no"], text["yes"]], horizontal=True)
        st_calls = 1 if calls_choice == text["yes"] else 0

        st_toxic = st.slider(text["f_toxic_circle"], 0, 3, value=1)

        respect_choice = st.radio(text["f_respect"], [text["no"], text["yes"]], horizontal=True)
        st_respect = 1 if respect_choice == text["yes"] else 0

    if st.button(text["btn_scan"], type="primary"):
        # Build the raw feature dictionary matching the training schema
        raw_input = {
            'Helps_In_Need_Score': st_help,
            'Keeps_Secrets_Rate': st_secrets,
            'Backbiting_Frequency': st_backbite,
            'Calls_Only_For_Help': st_calls,
            'Closest_Friends_Toxic_Count': st_toxic,
            'Respects_Others': st_respect
        }

        # One-hot encode categorical columns to match the original training X shape
        prof_list = ['Student', 'Software Engineer', 'Business Owner', 'Job Seeker']
        weak_list = ['Study/Work Focus', 'Emotional/Trusts Easily', 'Money Management', 'Anxiety/Stress']

        for p in prof_list:
            raw_input[f'User_Profession_{p}'] = 1 if user_prof == p else 0
        for w in weak_list:
            raw_input[f'User_Weakness_{w}'] = 1 if user_weak == w else 0

        # Align columns exactly to what the trained model expects
        input_df = pd.DataFrame([raw_input])[model_features]

        # Scale using the fitted StandardScaler
        input_scaled = scaler.transform(input_df)

        # Run the live KNN prediction
        prediction = knn_model.predict(input_scaled)[0]

        # Persist results for Tab 2
        st.session_state.prediction_label = prediction
        st.session_state.current_coords = (st_help, st_secrets)
        st.session_state.user_weak = user_weak
        st.session_state.audit_complete = True

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"<span class='status-true'>{text['res_true']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='status-fake'>{text['res_fake']}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2 — AI PERFECT FRIEND BLUEPRINT + 2D TRUST SPACE
# ============================================================
with tab2:
    st.markdown(f"### {text['bp_title']}")

    if st.session_state.audit_complete:
        u_weak = st.session_state.user_weak

        if selected_lang == "English":
            if u_weak == 'Emotional/Trusts Easily':
                shield_txt = ("Since you trust people too easily, do NOT share family secrets or "
                               "financial details for the first 6 months. Maintain strict emotional boundaries.")
                match_txt = ("An emotionally stable & grounded anchor who values logic and consistency "
                              "over sweet-talking behavior.")
            elif u_weak == 'Study/Work Focus':
                shield_txt = ("Your focus gets distracted easily. Avoid friends who constantly invite "
                               "you to waste time on social media or random outings.")
                match_txt = ("A highly ambitious, career-focused partner who actively pulls you into "
                              "deep study or work sessions.")
            elif u_weak == 'Money Management':
                shield_txt = ("You struggle with finance control. Keep clear records of money lending, "
                               "and set hard boundaries if a friend borrows frequently.")
                match_txt = ("A financially disciplined individual who respects commitments and talks "
                              "about productivity.")
            else:
                shield_txt = ("Since you face anxiety/stress, steer clear of friends who gossip heavily "
                               "or create unnecessary drama around you.")
                match_txt = "A calm, positive, and empathetic listener who reduces your baseline stress levels."

            flags_txt = (
                "1. Calls or texts you ONLY when they need an emergency favor.\n"
                "2. Their closest friends have a toxic/manipulative nature.\n"
                "3. High frequency of talking bad behind other people's back."
            )
        else:
            if u_weak == 'Emotional/Trusts Easily':
                shield_txt = ("चूंकि आप लोगों पर बहुत जल्दी भरोसा कर लेते हैं, इसलिए पहले 6 महीनों तक "
                               "पारिवारिक रहस्य या वित्तीय विवरण साझा न करें। सख्त भावनात्मक सीमाएं बनाएं।")
                match_txt = "एक भावनात्मक रूप से स्थिर और समझदार व्यक्ति जो मीठी बातों से ज्यादा सच्चाई को महत्व देता है।"
            elif u_weak == 'Study/Work Focus':
                shield_txt = ("आपका ध्यान आसानी से भटक जाता है। ऐसे दोस्तों से बचें जो आपको लगातार "
                               "सोशल मीडिया या फालतू घूमने के लिए बुलाते हैं।")
                match_txt = "एक अत्यधिक महत्वाकांक्षी, करियर-केंद्रित साथी जो आपको गहन अध्ययन या काम करने के लिए प्रेरित करे।"
            elif u_weak == 'Money Management':
                shield_txt = ("आप पैसों के प्रबंधन में संघर्ष करते हैं। किसी भी उधार का कड़ा हिसाब रखें, और "
                               "यदि कोई दोस्त बिना लौटाए बार-बार मदद मांगता है, तो मना करना सीखें।")
                match_txt = "एक वित्तीय रूप से अनुशासित व्यक्ति जो पैसों के वादों का सम्मान करता है और उत्पादकता की बातें करता है।"
            else:
                shield_txt = ("चूंकि आप तनाव/चिंता का सामना करते हैं, इसलिए उन दोस्तों से दूर रहें जो भारी "
                               "चुगली करते हैं या आपके आसपास फालतू का ड्रामा पैदा करते हैं।")
                match_txt = "एक शांत, सकारात्मक और सहानुभूति रखने वाला सुनने वाला साथी जो आपके मानसिक तनाव को कम करे।"

            flags_txt = (
                "1. केवल तभी कॉल या मेसेज करना जब उन्हें कोई जरूरी काम या स्वार्थ हो।\n"
                "2. उनके बाकी करीबी दोस्तों का सर्कल भी मतलबी या टॉक्सिक हो।\n"
                "3. दूसरों के पीठ पीछे लगातार बुराई या चुगली करने की आदत होना।"
            )

        # ---- Render Blueprint Cards ----
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class='glow-box-red'>
                <div class='suggestion-header'>{text['shield_plan']}</div>
                <p>{shield_txt}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='glow-box-purple'>
                <div class='suggestion-header'>{text['red_flags']}</div>
                <p style="white-space: pre-line;">{flags_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class='glow-box-green'>
                <div class='suggestion-header'>{text['ideal_match']}</div>
                <p>{match_txt}</p>
            </div>
            """, unsafe_allow_html=True)

        # ---- 2D Trust Coordinate Space Scatter Plot ----
        st.markdown("---")
        st.markdown(f"#### {text['plot_title']}")

        np.random.seed(42)
        pop_help = np.random.uniform(1.0, 10.0, 120)
        pop_secrets = np.random.uniform(10, 100, 120)
        pop_status = np.where(
            (pop_help * 4 + pop_secrets * 0.3) > 45,
            'True Friend Baseline',
            'Fake Friend Baseline'
        )
        plot_df = pd.DataFrame({
            'Help Score': pop_help,
            'Secret Keeping Rate (%)': pop_secrets,
            'Classification': pop_status
        })

        curr_h, curr_s = st.session_state.current_coords
        curr_p = st.session_state.prediction_label
        curr_label = '👉 YOUR AUDITED FRIEND (TRUE)' if curr_p == 1 else '👉 YOUR AUDITED FRIEND (RISK/FAKE)'

        user_friend_df = pd.DataFrame({
            'Help Score': [curr_h],
            'Secret Keeping Rate (%)': [curr_s],
            'Classification': [curr_label]
        })

        final_plot_df = pd.concat([plot_df, user_friend_df], ignore_index=True)
        final_plot_df['Marker Size'] = np.where(
            final_plot_df['Classification'].str.contains('YOUR'), 18, 6
        )

        fig = px.scatter(
            final_plot_df,
            x='Help Score',
            y='Secret Keeping Rate (%)',
            color='Classification',
            size='Marker Size',
            size_max=20,
            color_discrete_map={
                'True Friend Baseline': 'rgba(16, 185, 129, 0.25)',
                'Fake Friend Baseline': 'rgba(244, 63, 94, 0.25)',
                '👉 YOUR AUDITED FRIEND (TRUE)': '#10b981',
                '👉 YOUR AUDITED FRIEND (RISK/FAKE)': '#ef4444'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            legend_title_text='Classification'
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(text["no_data"])