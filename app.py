import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 網頁基本設定 (標題、圖示、排版)
# ==========================================
st.set_page_config(
    page_title="咖啡門市 AI 銷售預報系統",
    page_icon="☕",
    layout="centered"
)

# 網頁大標題
st.title("☕ 咖啡門市 AI 智慧備料與銷售預報系統")
st.markdown("這是基於 **XGBoost + GA 遺傳演算法** 優化後的黃金模型，輸入明日營運條件，即可一鍵估算最佳備料杯數。")
st.write("---")

# ==========================================
# 2. 安全載入 pickle 模型與標準化工具
# ==========================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load("coffee_xgb_model.pkl")
        scaler = joblib.load("coffee_scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ 找不到模型檔案或載入失敗！請確認 coffee_xgb_model.pkl 與 coffee_scaler.pkl 與本程式在同一個資料夾下。錯誤訊息: {e}")
        return None, None

model, scaler = load_models()

# ==========================================
# 3. 網頁前端介面：店長輸入區域 (Sidebar & Form)
# ==========================================
st.subheader("📊 請輸入明日營運與氣候預測條件：")

# 使用兩欄位排版，更整齊乾淨
col1, col2 = st.columns(2)

with col1:
    is_holiday_label = st.selectbox("📆 明天是國定假日或假期嗎？", ["工作日 (平日/週間)", "放假日 (週末/國定假日)"])
    is_holiday = 1.0 if is_holiday_label == "放假日 (週末/國定假日)" else 0.0
    
    temp = st.slider("🌡️ 明天預估氣溫 (°C)", min_value=0, max_value=40, value=25, step=1)
    staff = st.number_input("👥 明天預計排班員工數 (人)", min_value=1, max_value=15, value=4, step=1)

with col2:
    st.write("📢 明天預計舉辦的行銷活動：")
    buy1get1 = st.checkbox("🎟️ 買一送一 (Buy 1 Get 1 Free)", value=False)
    discount_20 = st.checkbox("💸 結帳八折大促銷 (20% OFF)", value=False)
    member_only = st.checkbox("👑 會員專屬優惠日", value=False)

# ==========================================
# 4. 後端 AI 計算與前端即時呈現
# ==========================================
st.write("---")

if model is not None and scaler is not None:
    # 建立模型訓練時的固定欄位結構（請與 X_opt.columns 完全對齊）
    # 注意：請確認順序與您在 Colab 訓練時的特徵順序完全一致
    feature_columns = [
        'Is_Holiday', 'Temperature_C', 'Staff_Count', 
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3',
        'Promotion_Buy1Get1', 'Promotion_Discount_20', 'Promotion_Member_Only'
    ]
    
    # 建立一個全零的單筆 DataFrame
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # 填入使用者勾選與輸入的資料
    input_data['Is_Holiday'] = float(is_holiday)
    input_data['Temperature_C'] = float(temp)
    input_data['Staff_Count'] = float(staff)
    
    # 促銷活動 One-Hot
    if buy1get1: input_data['Promotion_Buy1Get1'] = 1.0
    if discount_20: input_data['Promotion_Discount_20'] = 1.0
    if member_only: input_data['Promotion_Member_Only'] = 1.0
    
    # 時序特徵使用中位數模擬狀態 (或設為 0)
    input_data['Sales_Lag_1'] = 0.0
    input_data['Sales_Lag_2'] = 0.0
    input_data['Sales_Roll_Mean_3'] = 0.0
    
    # 5. 標準化 (僅標準化當初在訓練時需要被 scaler 縮放的五個欄位)
    scale_cols = ['Temperature_C', 'Staff_Count', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3']
    try:
        input_data[scale_cols] = scaler.transform(input_data[scale_cols])
        
        # 6. 一鍵預測
        if st.button("🚀 啟動 AI 大腦一鍵預測", type="primary"):
            prediction = model.predict(input_data)[0]
            
            # 呈現漂亮的儀表板視覺效果
            st.balloons() # 噴拉炮特效
            st.metric(label="🎯 明日預估最佳銷售杯數", value=f"{prediction:.1f} 杯")
            
            # 實務營運指南
            st.success("💡 **店長備料與排班指南建議**：")
            st.write(f"1. 請依 **{prediction:.1f} 杯** 的銷量規模準備鮮奶、咖啡豆與包材。")
            if is_holiday == 1.0:
                st.write("2. ⚠️ 明日逢放假日，屬於大賽道核心特徵，客流量極高，請務必確保物料充足、工讀生手速跟上。")
            if buy1get1:
                st.write("3. ⚠️ 辦理買一送一活動對庫存消耗速度極快，請特別注意當班備料原汁的存量。")
    except Exception as scaler_error:
        st.error(f"特徵對齊或標準化時出錯，請確認欄位名稱。錯誤：{scaler_error}")
