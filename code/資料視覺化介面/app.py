import streamlit as st
import pandas as pd
import plotly.express as px
from preprocess import preprocess_data, preprocess_data_for_model
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# 設定頁面配置
st.set_page_config(
    page_title="外送延遲分析系統",
    page_icon="🚚",
    layout="wide"
)

# 分頁結構
st.title("外送延遲分析系統")
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["外送延遲分析系統", "模型報表呈現", "單筆訂單 SHAP 解釋"])

with tab1:
    st.sidebar.header("資料上傳（分析系統）")
    uploaded_file1 = st.sidebar.file_uploader("上傳 CSV 檔案", type=['csv'], key="file1")
    if uploaded_file1 is not None:
        df = pd.read_csv(uploaded_file1)
        df_processed = preprocess_data(df)
        st.subheader("圖表選擇")
        chart_type = st.selectbox(
            "請選擇要顯示的圖表",
            ["延遲分布", "距離之訂單分布", "數值變數箱型圖", "類別變數分組長條圖", "星期分布長條圖",
             "短距離訂單中各延遲時間之訂單數", "短距離訂單中與延遲時長有關的因素", "短距離訂單中與延遲時長有關的因素2"]
        )
        # 延遲分布圖
        def plot_delay_distribution(df):
            delay_counts = df['delayed'].value_counts().reset_index()
            delay_counts.columns = ['Delayed', 'Count']
            fig = px.bar(
                delay_counts,
                x='Delayed',
                y='Count',
                title='Distribution of Delivery Delay',
                labels={
                    'Delayed': 'Delayed (0 = On Time, 1 = Delayed)',
                    'Count': 'Count'
                },
                color='Delayed',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_layout(
                showlegend=False,
                xaxis=dict(
                    tickmode='array',
                    ticktext=['On Time', 'Delayed'],
                    tickvals=[0, 1]
                ),
                template='plotly_white'
            )
            return fig
        # 距離分布圖
        def plot_distance_distribution(df):
            distance_bins = [0, 5, 10, 15, 20, float('inf')]
            distance_labels = ['0–5 km', '5–10 km', '10–15 km', '15–20 km', '20+ km']
            df['distance_group'] = pd.cut(df['distance_km'], bins=distance_bins, labels=distance_labels)
            group_counts = df['distance_group'].value_counts().sort_index().reset_index()
            group_counts.columns = ['Distance Group', 'Order Count']
            fig = px.bar(
                group_counts,
                x='Distance Group',
                y='Order Count',
                title='外送距離之訂單分布',
                labels={
                    'Distance Group': '距離範圍',
                    'Order Count': '訂單數量'
                },
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(
                template='plotly_white',
                xaxis_title='距離範圍',
                yaxis_title='訂單數量',
                showlegend=False
            )
            return fig
        # 箱型圖
        def plot_boxplots_separate(df):
            numeric_columns = ['Delivery_person_Age', 'Delivery_person_Ratings',
                              'multiple_deliveries', 'distance_km', 'Estimated_duration_minutes']
            plots = []
            for col in numeric_columns:
                fig = px.box(
                    df,
                    x='delayed',
                    y=col,
                    color='delayed',
                    color_discrete_map={0: 'skyblue', 1: 'salmon'},
                    points='outliers',
                    labels={
                        'delayed': 'Delayed (0 = On Time, 1 = Delayed)',
                        col: col
                    },
                    title=f'{col} vs. Delay'
                )
                fig.update_layout(
                    showlegend=False,
                    template='plotly_white'
                )
                plots.append(fig)
            return plots
        # 類別變數分組長條圖
        def plot_categorical_bars(df):
            if 'Order_Date' in df.columns:
                df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
                df['day_of_week'] = df['Order_Date'].dt.dayofweek.astype('category')
                df['day_of_week_name'] = df['Order_Date'].dt.day_name()
            if 'hour_ordered' not in df.columns:
                if 'Time_Orderd' in df.columns:
                    df['hour_ordered'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M', errors='coerce').dt.hour
                else:
                    df['hour_ordered'] = None
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
            df['time_segment'] = df['hour_ordered'].apply(assign_time_segment)
            categorical_vars = [
                'Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                'Type_of_vehicle', 'City', 'time_segment'
            ]
            plots = []
            for col in categorical_vars:
                plot_df = df.groupby([col, 'delayed']).size().reset_index(name='Count')
                plot_df['Delayed_Label'] = plot_df['delayed'].map({0: 'On Time', 1: 'Delayed'})
                fig = px.bar(
                    plot_df,
                    x=col,
                    y='Count',
                    color='Delayed_Label',
                    barmode='group',
                    color_discrete_map={'On Time': 'skyblue', 'Delayed': 'salmon'},
                    title=f'Delivery Delay by {col}',
                    labels={col: col, 'Count': 'Count', 'Delayed_Label': 'Delivery Status'}
                )
                fig.update_layout(
                    xaxis_title=col,
                    yaxis_title='Count',
                    legend_title='Delivery Status',
                    template='plotly_white'
                )
                fig.update_xaxes(tickangle=30)
                plots.append(fig)
            return plots
        # 星期分布長條圖
        def plot_day_of_week_bar(df):
            if 'Order_Date' in df.columns:
                df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
                df['day_of_week'] = df['Order_Date'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            plot_df = df.groupby(['day_of_week', 'delayed']).size().reset_index(name='Count')
            plot_df['Delayed_Label'] = plot_df['delayed'].map({0: 'On Time', 1: 'Delayed'})
            plot_df['day_of_week'] = pd.Categorical(plot_df['day_of_week'], categories=day_order, ordered=True)
            fig = px.bar(
                plot_df,
                x='day_of_week',
                y='Count',
                color='Delayed_Label',
                barmode='group',
                color_discrete_map={'On Time': 'skyblue', 'Delayed': 'salmon'},
                title='Delivery Delay by Day of Week',
                labels={'day_of_week': 'Day of Week', 'Count': 'Count', 'Delayed_Label': 'Delivery Status'}
            )
            fig.update_layout(
                xaxis_title='Day of Week',
                yaxis_title='Count',
                legend_title='Delivery Status',
                template='plotly_white'
            )
            return fig
        # 短距離訂單延遲時間分布圖
        def plot_short_distance_delay_distribution(df):
            from preprocess import preprocess_data_for_shortDistance_analysis
            short_orders = preprocess_data_for_shortDistance_analysis(df)

            # 手動指定 bins，確保與 matplotlib 一致
            delay_minutes = short_orders['delay_minutes']
            bins = np.histogram_bin_edges(delay_minutes, bins=30)
            bin_size = (bins[-1] - bins[0]) / (len(bins) - 1)

            # 計算平均與中位數
            mean_delay = delay_minutes.mean()
            median_delay = delay_minutes.median()

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=delay_minutes,
                xbins=dict(
                    start=bins[0],
                    end=bins[-1],
                    size=bin_size
                ),
                name='延遲時間分布',
                marker_color='skyblue',
                opacity=0.7
            ))

            # 取得 y 軸最大值，讓線條畫到底
            counts, _ = np.histogram(delay_minutes, bins=bins)
            y_max = counts.max()

            # 畫上平均延遲（紅線）
            fig.add_trace(go.Scatter(
                x=[mean_delay, mean_delay],
                y=[0, y_max],
                mode='lines',
                line=dict(color='red', dash='dash', width=3),
                name=f'平均延遲: {mean_delay:.2f} 分鐘'
            ))
            # 畫上中位數延遲（綠線）
            fig.add_trace(go.Scatter(
                x=[median_delay, median_delay],
                y=[0, y_max],
                mode='lines',
                line=dict(color='green', dash='dash', width=3),
                name=f'中位數延遲: {median_delay:.2f} 分鐘'
            ))

            fig.update_layout(
                title='短距離訂單延遲時間分布 (≤ 6 公里)',
                xaxis_title='延遲時間（分鐘）',
                yaxis_title='訂單數量',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    x=1,
                    y=1,
                    xanchor='right',
                    yanchor='top',
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='black',
                    borderwidth=1
                )
            )

            return fig
        # 短距離訂單中與延遲時長有關的因素 heatmap
        def plot_short_distance_delay_corr_heatmap(df):
            import plotly.express as px
            from preprocess import preprocess_data_for_shortDistance_analysis
            short_orders = preprocess_data_for_shortDistance_analysis(df)
            # 取得所有數值欄位
            numeric_cols = short_orders.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # 移除與 delay_minutes 直接相關的欄位
            filtered_numeric_cols = [col for col in numeric_cols if col not in ['Time_taken (min)', 'Estimated_duration_minutes']]
            # 計算相關係數
            filtered_corr = short_orders[filtered_numeric_cols].corr().round(2)
            # 只取與 delay_minutes 的相關性
            corr_delay = filtered_corr[['delay_minutes']].sort_values(by='delay_minutes', ascending=False)
            # 畫 heatmap，不加 title
            fig = px.imshow(
                corr_delay,
                text_auto=True,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                aspect='auto'
            )
            fig.update_layout(title='各數值欄位與延遲時間（delay_minutes）之相關係數')
            return fig
        def plot_short_distance_delay_categorical_boxplots(df):
            import plotly.express as px
            import pandas as pd
            from preprocess import preprocess_data_for_shortDistance_analysis
            short_orders = preprocess_data_for_shortDistance_analysis(df)

            # 時段分類函數
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

            # 轉換時段資訊
            short_orders['Time_Ordered_Hour'] = pd.to_datetime(short_orders['Time_Orderd'], errors='coerce').dt.hour
            short_orders['Time_Order_picked_Hour'] = pd.to_datetime(short_orders['Time_Order_picked'], errors='coerce').dt.hour
            short_orders['Ordered_time_segment'] = short_orders['Time_Ordered_Hour'].apply(assign_time_segment)
            short_orders['Picked_time_segment'] = short_orders['Time_Order_picked_Hour'].apply(assign_time_segment)

            categorical_vars = [
                'Order_Date', 'Ordered_time_segment', 'Picked_time_segment',
                'Weather_conditions', 'Road_traffic_density',
                'Type_of_order', 'Type_of_vehicle', 'City'
            ]

            figs = []
            for col in categorical_vars:
                if col in short_orders.columns:
                    fig = px.box(
                        short_orders,
                        x=col,
                        y='delay_minutes',
                        color=col,
                        title=f'Delay Minutes by {col}',
                        points='outliers',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_layout(
                        xaxis_title=col,
                        yaxis_title='delay_minutes',
                        showlegend=False
                    )
                    fig.update_xaxes(tickangle=90)
                    figs.append((col, fig))
            return figs

        # 根據選擇顯示對應的圖表
        if chart_type == "延遲分布":
            st.subheader("延遲分布")
            delay_fig = plot_delay_distribution(df_processed)
            st.plotly_chart(delay_fig, use_container_width=True)
        elif chart_type == "距離之訂單分布":
            st.subheader("距離之訂單分布")
            distance_fig = plot_distance_distribution(df_processed)
            st.plotly_chart(distance_fig, use_container_width=True)
        elif chart_type == "數值變數箱型圖":
            st.subheader("數值變數箱型圖")
            box_figs = plot_boxplots_separate(df_processed)
            cols = st.columns(2)
            for i, fig in enumerate(box_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "類別變數分組長條圖":
            st.subheader("類別變數分組長條圖")
            cat_figs = plot_categorical_bars(df_processed)
            cols = st.columns(2)
            for i, fig in enumerate(cat_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "星期分布長條圖":
            st.subheader("星期分布長條圖")
            day_fig = plot_day_of_week_bar(df_processed)
            st.plotly_chart(day_fig, use_container_width=True)
        elif chart_type == "短距離訂單中各延遲時間之訂單數":
            st.subheader("短距離訂單中各延遲時間之訂單數")
            short_distance_fig = plot_short_distance_delay_distribution(df)
            st.plotly_chart(short_distance_fig, use_container_width=True)
        elif chart_type == "短距離訂單中與延遲時長有關的因素":
            st.subheader("短距離訂單中與延遲時長有關的因素")
            heatmap_fig = plot_short_distance_delay_corr_heatmap(df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        elif chart_type == "短距離訂單中與延遲時長有關的因素2":
            st.subheader("短距離訂單中與延遲時長有關的因素2")
            boxplot_figs = plot_short_distance_delay_categorical_boxplots(df)
            cols = st.columns(2)
            for i, (col, fig) in enumerate(boxplot_figs):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請在側邊欄上傳 CSV 檔案以開始分析")

with tab2:
    st.sidebar.header("資料上傳（模型報表）")
    uploaded_file2 = st.sidebar.file_uploader("上傳模型分析用 CSV", type=['csv'], key="file2")
    xgb_model = joblib.load("models/xgb_model.joblib")
    if uploaded_file2 is not None:
        df = pd.read_csv(uploaded_file2)
        df_processed = preprocess_data(df)
        X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features = preprocess_data_for_model(df_processed)
        preprocessor.fit(X_train)
        X_transformed = preprocessor.transform(X_test)
        # 取出 pipeline 中的 XGBoost 分類器
        model = xgb_model
        if hasattr(xgb_model, 'named_steps') and 'classifier' in xgb_model.named_steps:
            model = xgb_model.named_steps['classifier']
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_transformed)
            feature_names = preprocessor.get_feature_names_out()
            def plot_shap_summary_bar(shap_values, X_transformed, feature_names):
                st.subheader("SHAP Summary Bar Plot (XGBoost)")
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False)
                st.pyplot(fig)
            def plot_shap_dependence_original(shap_values, X_transformed, feature_names, feature, X_test):
                st.subheader(f"SHAP Dependence Plot - {feature}")
                import pandas as pd
                if hasattr(X_transformed, 'toarray'):
                    X_shap_plot = pd.DataFrame(X_transformed.toarray(), columns=feature_names)
                else:
                    X_shap_plot = pd.DataFrame(X_transformed, columns=feature_names)
                # 根據選擇覆蓋原始資料
                if feature == 'num__distance_km' and 'distance_km' in X_test.columns:
                    X_shap_plot[feature] = X_test['distance_km'].reset_index(drop=True)
                elif feature == 'num__Delivery_person_Age' and 'Delivery_person_Age' in X_test.columns:
                    X_shap_plot[feature] = X_test['Delivery_person_Age'].reset_index(drop=True)
                # 類別型特徵不需覆蓋
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.dependence_plot(
                    ind=feature,
                    shap_values=shap_values.values if hasattr(shap_values, 'values') else shap_values,
                    features=X_shap_plot,
                    feature_names=feature_names,
                    interaction_index=None,
                    ax=ax
                )
                st.pyplot(fig)
            shap_chart_type = st.selectbox(
                "請選擇要顯示的 SHAP 圖表",
                ["Summary Bar Plot", "Dependence Plot"]
            )
            allowed_features = ["num__Delivery_person_Age", "num__distance_km", "cat__Road_traffic_density_Low"]
            if shap_chart_type == "Summary Bar Plot":
                plot_shap_summary_bar(shap_values, X_transformed, feature_names)
            elif shap_chart_type == "Dependence Plot":
                feature = st.selectbox("請選擇要分析的特徵", allowed_features, key="dep_raw")
                plot_shap_dependence_original(shap_values, X_transformed, feature_names, feature, X_test)
        except Exception as e:
            st.error(f"SHAP 分析時發生錯誤：{e}")
    else:
        st.info("請在側邊欄上傳模型分析用 CSV 檔案以顯示 SHAP 分析圖表")

with tab3:
    st.sidebar.header("資料上傳（單筆訂單分析）")
    uploaded_file3 = st.sidebar.file_uploader("上傳欲分析的訂單 CSV", type=['csv'], key="file3")
    xgb_model = joblib.load("models/xgb_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    if uploaded_file3 is not None:
        df = pd.read_csv(uploaded_file3)
        df_processed = preprocess_data(df)
        from preprocess import preprocess_data_for_inference
        X = preprocess_data_for_inference(df_processed)
        # 直接用訓練時的 preprocessor 做 transform，不要 fit
        X_transformed = preprocessor.transform(X)
        # 取出 pipeline 中的 XGBoost 分類器
        model = xgb_model
        if hasattr(xgb_model, 'named_steps') and 'classifier' in xgb_model.named_steps:
            model = xgb_model.named_steps['classifier']
        try:
            explainer = shap.Explainer(model)
            feature_names = preprocessor.get_feature_names_out()
            sample_orders = X.reset_index(drop=True)
            results = []
            for i in range(len(sample_orders)):
                raw_order = sample_orders.iloc[[i]]
                order_transformed = preprocessor.transform(raw_order)
                prob = model.predict_proba(order_transformed)[:, 1][0]

                # 延遲風險分類
                if prob >= 0.7:
                    delay_risk_levels = "高"
                elif prob >= 0.3:
                    delay_risk_levels = "中"
                else:
                    delay_risk_levels = "低"

                # SHAP 分析
                shap_values_i = explainer(order_transformed)
                shap_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values_i.values[0]
                }).sort_values(by='shap_value', ascending=False).head(5)
                results.append({
                    'delay_prob_levels': delay_risk_levels,
                    'top_factors': shap_df.reset_index(drop=True)
                })
                
            # 呈現所有結果，每筆用expander包起來
            risk_levels = ["無", "全部", "高", "中", "低"]
            selected_risk = st.selectbox("選擇要查看的延遲風險等級：", risk_levels, key="risk_level_select", index=0)
            if selected_risk != "無":
                # 依選擇的風險等級篩選資料
                filtered_idx = [i for i, res in enumerate(results) if selected_risk == "全部" or res['delay_prob_levels'] == selected_risk]
                filtered_X = sample_orders.iloc[filtered_idx].reset_index(drop=True)
                st.subheader("符合條件的訂單資料：")
                st.dataframe(filtered_X, use_container_width=True, height=350)
                # index 輸入查詢
                if len(filtered_X) > 0:
                    st.markdown("---")
                    idx_range = f"0 ~ {len(filtered_X)-1}"
                    query_idx_str = st.text_input(f"請輸入要查詢的 index（{idx_range}）：", value="", key="query_idx")
                    if query_idx_str.isdigit():
                        query_idx = int(query_idx_str)
                        if 0 <= query_idx < len(filtered_X):
                            real_idx = filtered_idx[query_idx]
                            res = results[real_idx]
                            st.markdown(f"### 訂單延遲風險 : {res['delay_prob_levels']}")
                            factor_descriptions = [
                                f"第{i+1}大延誤因子：{row['feature']}" for i, row in res['top_factors'].iterrows()
                            ]
                            desc_df = pd.DataFrame({'延誤因子': factor_descriptions})
                            st.dataframe(desc_df, use_container_width=True, hide_index=True, column_order=None)
                        else:
                            st.warning(f"請輸入範圍內的 index（{idx_range}）")
                    elif query_idx_str != "":
                        st.warning("請輸入有效的整數 index")

        except Exception as e:
            st.error(f"SHAP 單筆分析時發生錯誤：{e}")
    else:
        st.info("請在側邊欄上傳欲分析的訂單 CSV 檔案")