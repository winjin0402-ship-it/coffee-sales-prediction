import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 網頁基本設定
# ==========================================
st.set_page_config(
    page_title="咖啡門市 AI 銷售預報系統",
    page_icon="☕",
    layout="centered"
)

st.title("☕ 咖啡門市 AI 智慧備料與銷售預報系統")
st.markdown("這是基於 **XGBoost + GA 遺傳演算法** 優化後的黃金模型。")
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
        return None, None

model, scaler = load_models()

if model is None or scaler is None:
    st.error("❌ 讀取模型組件 (.pkl) 時發生版本不相容或找不到檔案。請點選右下角 Manage App 查看詳細錯誤日誌。")

# ==========================================
# 3. 網頁前端介面：店長輸入區域
# ==========================================
st.subheader("📊 請輸入明日營運與氣候預測條件：")
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

st.write("---")

# ==========================================
# 4. 後端 AI 計算與安全防禦
# ==========================================
if model is not None and scaler is not None:
    # 建立與您 XGBoost 模型完全對齊的 9 個核心特徵欄位
    feature_columns = [
        'Is_Holiday', 'Temperature_C', 'Staff_Count', 
        'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3',
        'Promotion_Buy1Get1', 'Promotion_Discount_20', 'Promotion_Member_Only'
    ]
    
    # 建立數據
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    input_data['Is_Holiday'] = float(is_holiday)
    input_data['Temperature_C'] = float(temp)
    input_data['Staff_Count'] = float(staff)
    
    if buy1get1: input_data['Promotion_Buy1Get1'] = 1.0
    if discount_20: input_data['Promotion_Discount_20'] = 1.0
    if member_only: input_data['Promotion_Member_Only'] = 1.0

    # 觸發預測按鈕
    if st.button("🚀 啟動 AI 大腦一鍵預測", type="primary"):
        try:
            # 🌟 [超級安全牌修正]：如果 scaler 因為版本問題不認得欄位名稱，則強制抽取數值進行轉換
            scale_cols = ['Temperature_C', 'Staff_Count', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3']
            
            # 先複製一份，避免改動原始 DataFrame 結構
            scaled_features = input_data[scale_cols].values
            scaled_transformed = scaler.transform(scaled_features)
            
            # 把標準化後的數值塞回去
            input_data[scale_cols] = scaled_transformed
            
            # 丟進全場冠軍 XGBoost 進行預測
            prediction = model.predict(input_data)[0]
            
            # 成功噴拉炮呈現
            st.balloons()
            st.metric(label="🎯 明日預估最佳銷售杯數", value=f"{prediction:.1f} 杯")
            
            st.success("💡 **店長備料與排班指南建議**：")
            st.write(f"1. 請依 **{prediction:.1f} 杯** 的銷量規模準備鮮奶、咖啡豆與包材。")
            if is_holiday == 1.0:
                st.write("2. ⚠️ 明日逢放假日，客流量極高，請務必確保物料充足、工讀生手速跟上。")
            if buy1get1:
                st.write("3. ⚠️ 辦理買一送一活動對庫存消耗速度極快，請特別注意當班備料存量。")
                
        except Exception as run_err:
            st.error(f"⚠️ 模型在進行計算時發生錯誤，這通常是 .pkl 檔案欄位不對齊導致。錯誤詳情：{run_err}")
