import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. 網頁基本設定 (標題、網頁圖示、版面配置)
# ==========================================
st.set_page_config(
    page_title="咖啡門市 AI 銷售預報系統",
    page_icon="☕",
    layout="centered"
)

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
        return None, None

model, scaler = load_models()

if model is None or scaler is None:
    st.error("❌ 讀取模型組件 (.pkl) 時發生版本不相容或找不到檔案。請點選右下角 Manage App 查看詳細錯誤日誌。")

# ==========================================
# 3. 網頁前端介面：店長輸入區域 (兩欄式排版)
# ==========================================
st.subheader("📊 請輸入明日營運與氣候預測條件：")

col1, col2 = st.columns(2)

with col1:
    is_holiday_label = st.selectbox("📆 明天是國定假日或假期嗎？", ["工作日 (平日/週間)", "放假日 (週末/國定假日)"])
    is_holiday = 1.0 if is_holiday_label == "放假日 (週末/國定假日)" else 0.0
    
    weather_type = st.selectbox("🌤️ 明天預估天氣狀況", ["晴天 (Sunny)", "陰天 (Cloudy)", "多雲 (Overcast)", "下雨天 (Rainy)"])
    
    temp = st.slider("🌡️ 明天預估氣溫 (°C)", min_value=0, max_value=40, value=25, step=1)
    staff = st.number_input("👥 明天預計排班員工數 (人)", min_value=1, max_value=15, value=4, step=1)
    
    ad_imp = st.number_input("📈 明天預計投放廣告曝光量 (次)", min_value=0, value=5000, step=500)

with col2:
    st.write("📢 明天預計舉辦的行銷活動：")
    buy1get1 = st.checkbox("🎟️ 買一送一 (Buy 1 Get 1 Free)", value=False)
    discount_20 = st.checkbox("💸 結帳八折大促銷 (20% OFF)", value=False)
    member_only = st.checkbox("👑 會員專屬優惠日", value=False)
    
    st.write("💰 價格策略微調 (若無調整保持預設即可)：")
    our_price = st.number_input("☕ 自家產品平均定價 (元)", min_value=10, max_value=300, value=65)
    comp_price = st.number_input("🏪 隔壁對手平均定價 (元)", min_value=10, max_value=300, value=60)

st.write("---")

# ==========================================
# 4. 後端 AI 計算與 31 特徵精準對齊 (完美根除 Mismatch 報錯)
# ==========================================
if model is not None and scaler is not None:
    # 🌟 核心修復：31個特徵順序必須與模型在 Colab 訓練時完全一致，一字不差！
    feature_columns = [
        'Temperature_C', 'Is_Holiday', 'Staff_Count', 'Price', 'Ad_Impressions', 'Competitor_Price',
        'Hour', 'Month', 'Is_Weekend', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3',
        'Rain_and_Promo', 'Weekend_Afternoon', 'Is_Hot_Sunny',
        'Weather_Cloudy', 'Weather_Overcast', 'Weather_Rainy', 'Weather_Sunny',
        'Time_Slot_Afternoon', 'Time_Slot_Evening', 'Time_Slot_Morning', 'Time_Slot_Night',
        'Promotion_Buy1Get1', 'Promotion_Discount_20', 'Promotion_Member_Only',
        'Coffee_Type_Americano', 'Coffee_Type_Cappuccino', 'Coffee_Type_Espresso', 'Coffee_Type_Latte', 'Coffee_Type_Mocha'
    ]
    
    # 建立單筆資料 DataFrame (🌟 index 已修正為 [0])
    input_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # 填入使用者輸入的基礎特徵
    input_data['Temperature_C'] = float(temp)
    input_data['Is_Holiday'] = float(is_holiday)
    input_data['Staff_Count'] = float(staff)
    input_data['Price'] = float(our_price)
    input_data['Ad_Impressions'] = float(ad_imp)
    input_data['Competitor_Price'] = float(comp_price)
    
    # 填入時間與循環模擬預設值
    input_data['Hour'] = 14.0       # 模擬下午黃金時段
    input_data['Month'] = 6.0        # 模擬年中旺季
    input_data['Is_Weekend'] = 1.0 if is_holiday == 1.0 else 0.0
    
    # 天氣狀況 One-Hot 填入
    if "Cloudy" in weather_type: input_data['Weather_Cloudy'] = 1.0
    if "Overcast" in weather_type: input_data['Weather_Overcast'] = 1.0
    if "Rainy" in weather_type: input_data['Weather_Rainy'] = 1.0
    if "Sunny" in weather_type: input_data['Weather_Sunny'] = 1.0
    
    # 促銷活動 One-Hot 填入
    if buy1get1: input_data['Promotion_Buy1Get1'] = 1.0
    if discount_20: input_data['Promotion_Discount_20'] = 1.0
    if member_only: input_data['Promotion_Member_Only'] = 1.0
    
    # 時段固定模擬下午茶 (對應多數爆單場景)
    input_data['Time_Slot_Afternoon'] = 1.0
    
    # 咖啡品類平均分配權重 (設定代表性主力拿鐵咖啡)
    input_data['Coffee_Type_Latte'] = 1.0
    
    # 進階交互特徵邏輯計算
    input_data['Rain_and_Promo'] = 1.0 if ("Rainy" in weather_type and (buy1get1 or discount_20 or member_only)) else 0.0
    input_data['Weekend_Afternoon'] = 1.0 if (input_data['Is_Weekend'] == 1.0) else 0.0
    input_data['Is_Hot_Sunny'] = 1.0 if (temp > 25 and "Sunny" in weather_type) else 0.0

    # 觸發預測按鈕
    if st.button("🚀 啟動 AI 大腦一鍵預測", type="primary"):
        try:
            # 🌟 標準化縮放修正
            scale_cols = ['Temperature_C', 'Staff_Count', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Roll_Mean_3']
            
            # 使用 values 進行純矩陣標準化，繞過版本名稱檢查
            scaled_features = input_data[scale_cols].values
            scaled_transformed = scaler.transform(scaled_features)
            input_data[scale_cols] = scaled_transformed
            
            # 丟進全場冠軍 XGBoost 進行預測
            prediction = model.predict(input_data)[0]
            
            # 成功噴拉炮呈現
            st.balloons()
            st.metric(label="🎯 明日預估最佳銷售杯數", value=f"{prediction:.1f} 杯")
            
            st.success("💡 **店長備料與排班指南建議**：")
            st.write(f"1. 請依 **{prediction:.1f} 杯** 的銷量規模準備鮮奶、咖啡豆與包材。")
            if is_holiday == 1.0:
                st.write("2. ⚠️ 明日逢放假日，此模型中權重佔比高達 47.44%，客流量極大，物料與人力請務必一步到位。")
            if buy1get1:
                st.write("3. ⚠️ 辦理買一送一活動對庫存消耗速度極快，請特別注意當班備料存量。")
                
        except Exception as run_err:
            st.error(f"⚠️ 模型在進行計算時發生錯誤，錯誤詳情：{run_err}")
