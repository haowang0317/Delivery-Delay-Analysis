import pandas as pd

def preprocess_data(df):
    # 1. 資料清理
    df_cleaned = df.dropna(subset=['multiple_deliveries'])
    df_cleaned = df_cleaned[~((df_cleaned['Festival'] == 'Yes') | (df_cleaned['Festival'].isna()))]
    
    # 2. 建立目標變數
    df_cleaned['delayed'] = (df_cleaned['Time_taken (min)'] > df_cleaned['Estimated_duration_minutes']).astype(int)

    return df_cleaned

def preprocess_data_for_model(df):
    """
    將原始資料 df 進行前處理，回傳 X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features。
    """
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    # Step 1: 移除延遲定義用的欄位
    X = df.drop(columns=['delayed', 'order_datetime', 'Time_taken (min)', 'Estimated_duration_minutes'])
    y = df['delayed']
    # Step 2: 時間欄位轉換與萃取特徵
    X['Order_Date'] = pd.to_datetime(X['Order_Date'], format='%d-%m-%Y')
    X['hour_ordered'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce').dt.hour
    X['hour_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce').dt.hour
    X['day_of_week'] = X['Order_Date'].dt.dayofweek.astype('category')
    # Step 3: 時段分類
    def assign_time_segment(hour):
        if pd.isnull(hour):
            return 'unknown'
        elif 8 <= hour <= 11:
            return 'morning_noon'
        elif 12 <= hour <= 16:
            return 'afternoon'
        elif hour >= 17:
            return 'evening'
        else:
            return 'other'
    X['time_segment'] = X['hour_ordered'].apply(assign_time_segment)
    # Step 4: 移除原始時間欄位
    X = X.drop(columns=['Time_Orderd', 'Time_Order_picked'])
    # Step 5: 特徵分類
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'datetime64']).columns.tolist()
    # Step 6: 建立前處理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    # Step 7: 切分訓練與測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features

def preprocess_data_for_inference(df):
    """
    將原始資料 df 進行前處理，回傳 X。
    """
    # Step 1: 移除延遲定義用的欄位
    X = df.drop(columns=['delayed', 'order_datetime', 'Time_taken (min)', 'Estimated_duration_minutes'])
    # Step 2: 時間欄位轉換與萃取特徵
    X['Order_Date'] = pd.to_datetime(X['Order_Date'], format='%d-%m-%Y')
    X['hour_ordered'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce').dt.hour
    X['hour_picked'] = pd.to_datetime(df['Time_Order_picked'], format='%H:%M', errors='coerce').dt.hour
    X['day_of_week'] = X['Order_Date'].dt.dayofweek.astype('category')
    # Step 3: 時段分類
    def assign_time_segment(hour):
        if pd.isnull(hour):
            return 'unknown'
        elif 8 <= hour <= 11:
            return 'morning_noon'
        elif 12 <= hour <= 16:
            return 'afternoon'
        elif hour >= 17:
            return 'evening'
        else:
            return 'other'
    X['time_segment'] = X['hour_ordered'].apply(assign_time_segment)
    # Step 4: 移除原始時間欄位
    X = X.drop(columns=['Time_Orderd', 'Time_Order_picked'])
    return X

def preprocess_data_for_shortDistance_analysis(df):
    df_cleaned = df.dropna(subset=['multiple_deliveries'])
    df_cleaned = df_cleaned[~((df_cleaned['Festival'] == 'Yes') | (df_cleaned['Festival'].isna()))]
    df_cleaned = df_cleaned.drop(columns=['Festival'])

    df_cleaned['delay_minutes'] = df_cleaned['Time_taken (min)'] - df_cleaned['Estimated_duration_minutes']


    short_distance = df_cleaned[df_cleaned['distance_km'] <= 6]
    return short_distance

    
    