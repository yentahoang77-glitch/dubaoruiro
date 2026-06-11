import streamlit as nn
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import io

# 1) Cấu hình trang đầu tiên
nn.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch gian lận tại Agribank",
    page_icon="🛡️"
)

# 2) Các hàm cache dùng chung
@nn.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo khả năng hash của Streamlit cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        return None

# Định nghĩa danh sách các biến đặc trưng dựa theo notebook và dataset
FEATURE_COLS = [f'X_{i}' for i in range(1, 15)]
TARGET_COL = 'default'

# 3) SIDEBAR - VÙNG CẤU HÌNH
with nn.sidebar:
    nn.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện
    uploaded_file = nn.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV hoặc Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Tải lên tệp chứa các biến đặc trưng X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    nn.divider()
    nn.subheader("Tham số mô hình AI")
    nn.caption("Thuật toán: RandomForestClassifier")
    
    # Các siêu tham số dựa theo mô hình trong notebook
    n_estimators = nn.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    max_depth = nn.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, 
        max_value=30, 
        value=10, 
        step=1,
        help="Độ sâu tối đa của mỗi cây quyết định (Trống/None tương đương kiểm soát bằng slider)."
    )
    
    criterion = nn.selectbox(
        "Tiêu chí đánh giá (criterion)",
        options=["gini", "entropy", "log_loss"],
        index=0,
        help="Hàm đo lường chất lượng phân tách."
    )
    
    with nn.expander("Tham số nâng cao"):
        random_state = nn.number_input(
            "Mầm ngẫu nhiên (random_state)", 
            value=42, 
            step=1,
            help="Đảm bảo tính tái lập của kết quả huấn luyện."
        )
        test_size = nn.slider(
            "Tỷ lệ tập kiểm tra (test_size)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3, 
            step=0.05,
            help="Tỷ lệ phần trăm dữ liệu dùng để đánh giá mô hình."
        )

    nn.divider()
    # Nút bấm hành động kích hoạt huấn luyện duy nhất
    train_clicked = nn.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu phân tách dữ liệu và huấn luyện mô hình Random Forest."
    )

# 4) HEADER - VÙNG ĐỊNH HƯỚNG
nn.title("🛡️ Hệ thống Dự báo Rủi ro & Phát hiện Gian lận")
nn.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, huấn luyện mô hình máy học Random Forest và dự báo tự động trạng thái rủi ro (default).")

if uploaded_file is None:
    nn.info("👋 Chào mừng bạn! Vui lòng tải tệp dữ liệu (.csv hoặc .xlsx) ở thanh bên (Sidebar) để bắt đầu.")
    nn.stop()

# Đọc dữ liệu khi đã upload
file_bytes = uploaded_file.getvalue()
df_raw = load_data(file_bytes, uploaded_file.name)

if df_raw is None:
    nn.error("❌ Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại định dạng file.")
    nn.stop()

# Kiểm tra sự hiện diện của các cột bắt buộc
missing_features = [col for col in FEATURE_COLS if col not in df_raw.columns]
if missing_features or (TARGET_COL not in df_raw.columns):
    nn.error(f"❌ Cấu trúc tệp dữ liệu không hợp lệ.")
    if missing_features:
        nn.write(f"Thiếu các cột đặc trưng: {missing_features}")
    if TARGET_COL not in df_raw.columns:
        nn.write(f"Thiếu cột biến mục tiêu: '{TARGET_COL}'")
    nn.stop()

nn.caption(f"📁 Đang sử dụng tệp dữ liệu: `{uploaded_file.name}`")
nn.divider()

# 5) KHỐI XỬ LÝ HUẤN LUYỆN (Chạy khi bấm nút và lưu vào session_state)
if train_clicked:
    with nn.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình..."):
        X = df_raw[FEATURE_COLS]
        y = df_raw[TARGET_COL]
        
        # Phân chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Khởi tạo và huấn luyện mô hình
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Dự đoán đánh giá
        y_pred = model.predict(X_test)
        
        # Tính toán các chỉ số kiểm định
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Lưu vào trạng thái phiên làm việc (session_state)
        nn.session_state["trained_model"] = model
        nn.session_state["metrics"] = metrics
        nn.session_state["feature_names"] = FEATURE_COLS
        nn.success("🎉 Huấn luyện mô hình thành công! Hãy chuyển sang các tab bên dưới để xem kết quả.")

# 6) TẠO CÁC TABS NỘI DUNG & KẾT QUẢ
tab1, tab2, tab3, tab4 = nn.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Sử dụng mô hình"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tab1:
    nn.subheader("Thống kê cấu trúc tập dữ liệu")
    col_m1, col_m2, col_m3 = nn.columns(3)
    col_m1.metric("Số lượng bản ghi (Dòng)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số lượng trường (Cột)", f"{df_raw.shape[1]}")
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m3.metric("Dung lượng tệp", f"{file_size_mb:.2f} MB")
    
    nn.markdown("### 📋 Xem trước dữ liệu thô (5 dòng đầu)")
    nn.dataframe(df_raw.head(5), use_container_width=True)
    
    nn.markdown("### 🔢 Thống kê mô tả các biến mô hình")
    # Chỉ thống kê mô tả các biến được đưa vào mô hình theo quy tắc chung
    cols_to_desc = FEATURE_COLS + [TARGET_COL]
    nn.dataframe(df_raw[cols_to_desc].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tab2:
    nn.subheader("Biểu đồ phân phối các biến quan trọng")
    
    # Ưu tiên hiển thị biến mục tiêu trước
    col_g1, col_g2 = nn.columns(2)
    with col_g1:
        target_counts = df_raw[TARGET_COL].value_counts().reset_index()
        target_counts.columns = [TARGET_COL, 'Số lượng']
        target_counts[TARGET_COL] = target_counts[TARGET_COL].astype(str)
        fig_target = px.bar(
            target_counts, x=TARGET_COL, y='Số lượng',
            title=f"Phân phối biến mục tiêu ({TARGET_COL})",
            color=TARGET_COL, color_discrete_sequence=px.colors.qualitative.Set2
        )
        nn.plotly_chart(fig_target, use_container_width=True)
        
    with col_g2:
        # Biểu đồ biến đặc trưng quan trọng đầu tiên X_1
        fig_x1 = px.histogram(
            df_raw, x="X_1", color=TARGET_COL,
            title="Phân phối của biến đặc trưng X_1",
            marginal="box", barmode="overlay"
        )
        nn.plotly_chart(fig_x1, use_container_width=True)
        
    col_g3, col_g4 = nn.columns(2)
    with col_g3:
        fig_x2 = px.histogram(
            df_raw, x="X_2", color=TARGET_COL,
            title="Phân phối của biến đặc trưng X_2",
            marginal="box", barmode="overlay"
        )
        nn.plotly_chart(fig_x2, use_container_width=True)
        
    with col_g4:
        # Cho phép người dùng tự chọn biến bổ sung nếu danh sách biến quá nhiều
        selected_var = nn.selectbox("Chọn biến đặc trưng khác để trực quan hóa:", options=FEATURE_COLS[2:])
        fig_custom = px.histogram(
            df_raw, x=selected_var, color=TARGET_COL,
            title=f"Phân phối biến tùy chọn {selected_var}",
            marginal="violin", barmode="overlay"
        )
        nn.plotly_chart(fig_custom, use_container_width=True)

# --- TAB 3: KẾT QUẢ HUÂN LUYỆN & KIỂM ĐỊNH MÔ HÌNH ---
with tab3:
    nn.subheader("Chỉ số hiệu năng kiểm định mô hình")
    
    if "metrics" not in nn.session_state:
        nn.info("💡 Vui lòng thiết lập cấu hình ở Sidebar và nhấn nút '🚀 Huấn luyện mô hình' để xem kết quả kiểm định.")
    else:
        metrics = nn.session_state["metrics"]
        
        # Hiển thị các chỉ số vô hướng chính
        c_m1, c_m2, c_m3, c_m4 = nn.columns(4)
        c_m1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        c_m2.metric("Precision (Lớp 1)", f"{metrics['precision']:.4f}")
        c_m3.metric("Recall (Lớp 1)", f"{metrics['recall']:.4f}")
        c_m4.metric("F1-Score (Lớp 1)", f"{metrics['f1']:.4f}")
        
        nn.divider()
        col_res1, col_res2 = nn.columns(2)
        
        with col_res1:
            nn.markdown("#### 📊 Ma trận nhầm lẫn (Confusion Matrix)")
            cm_df = pd.DataFrame(
                metrics["cm"], 
                index=["Thực tế: 0", "Thực tế: 1"], 
                columns=["Dự báo: 0", "Dự báo: 1"]
            )
            nn.dataframe(cm_df, use_container_width=True)
            
            # Trực quan hóa ma trận nhầm lẫn dạng Heatmap bằng Plotly
            fig_cm = px.imshow(
                metrics["cm"],
                text_auto=True,
                labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
                x=['Nhãn 0', 'Nhãn 1'],
                y=['Nhãn 0', 'Nhãn 1'],
                color_continuous_scale="Blues"
            )
            nn.plotly_chart(fig_cm, use_container_width=True)
            
        with col_res2:
            nn.markdown("#### 📋 Báo cáo chi tiết (Classification Report)")
            report_df = pd.DataFrame(metrics["report"]).transpose()
            nn.dataframe(report_df.style.format(precision=4), use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH ---
with tab4:
    nn.subheader("Dự báo dữ liệu giao dịch mới")
    
    if "trained_model" not in nn.session_state:
        nn.info("💡 Vui lòng huấn luyện mô hình thành công tại Sidebar trước khi thực hiện dự báo dữ liệu mới.")
    else:
        model = nn.session_state["trained_model"]
        
        mode = nn.radio(
            "Chọn phương thức nhập dữ liệu đầu vào:",
            options=["Nhập trực tiếp qua form", "Tải lên tệp danh sách (X_new)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP
        if mode == "Nhập trực tiếp qua form":
            nn.markdown("✍️ *Nhập các thông số kỹ thuật của giao dịch cần kiểm tra:*")
            
            with nn.form("single_prediction_form"):
                # Tạo lưới nhập liệu gồm 3 cột để giao diện gọn gàng
                form_cols = nn.columns(3)
                input_data = {}
                
                for idx, col_name in enumerate(FEATURE_COLS):
                    col_target = form_cols[idx % 3]
                    # Lấy giá trị trung vị từ tập dữ liệu thô làm mặc định
                    default_val = float(df_raw[col_name].median())
                    min_val = float(df_raw[col_name].min())
                    max_val = float(df_raw[col_name].max())
                    
                    with col_target:
                        input_data[col_name] = nn.number_input(
                            f"Nhập {col_name}",
                            min_value=min_val - 10.0,
                            max_value=max_val + 10.0,
                            value=default_val,
                            format="%.6f"
                        )
                
                submit_pred = nn.form_submit_button("🔍 Tiến hành dự báo rủi ro")
                
            if submit_pred:
                # Chuyển đổi dữ liệu nhập vào thành DataFrame đúng định dạng cột lúc train
                input_df = pd.DataFrame([input_data])[FEATURE_COLS]
                
                prediction = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                nn.divider()
                nn.markdown("### 🔔 Kết quả kiểm tra")
                if prediction == 1:
                    nn.error(f"🚨 CẢNH BÁO: Giao dịch có nguy cơ GIAN LẬN / MẶC ĐỊNH cao! (Nhãn: 1)")
                else:
                    nn.success(f"✅ AN TOÀN: Giao dịch có mức độ rủi ro thấp. (Nhãn: 0)")
                    
                col_p1, col_p2 = nn.columns(2)
                col_p1.metric("Xác suất An toàn (Lớp 0)", f"{probabilities[0]*100:.2f}%")
                col_p2.metric("Xác suất Rủi ro (Lớp 1)", f"{probabilities[1]*100:.2f}%")

        # CHẾ ĐỘ 2: TẢI FILE HÀNG LOẠT
        elif mode == "Tải lên tệp danh sách (X_new)":
            nn.markdown("📂 *Tải lên file Excel hoặc CSV chứa các cột từ `X_1` đến `X_14` để dự báo hàng loạt.*")
            
            batch_file = nn.file_uploader(
                "Tải lên tệp dữ liệu kiểm tra mới",
                type=["csv", "xlsx", "xls"],
                key="batch_uploader"
            )
            
            if batch_file is not None:
                batch_bytes = batch_file.getvalue()
                df_batch = load_data(batch_bytes, batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra tính khớp của các cột đặc trưng
                    missing_batch_cols = [c for c in FEATURE_COLS if c not in df_batch.columns]
                    
                    if missing_batch_cols:
                        nn.error(f"❌ Tệp tải lên thiếu các cột bắt buộc sau để dự báo: {missing_batch_cols}")
                    else:
                        # Trích xuất đúng tập biến đầu vào
                        X_batch = df_batch[FEATURE_COLS]
                        
                        # Dự báo hàng loạt
                        batch_preds = model.predict(X_batch)
                        batch_probs = model.predict_proba(X_batch)[:, 1] # Xác suất lớp rủi ro (1)
                        
                        # Tạo DataFrame kết quả kết hợp dữ liệu gốc
                        df_result = df_batch.copy()
                        df_result["Dự_Báo_Default"] = batch_preds
                        df_result["Xác_Suất_Rủi_Ro"] = batch_probs
                        
                        nn.success(f"📊 Đã xử lý dự báo thành công cho {df_result.shape[0]} dòng dữ liệu.")
                        
                        # Thống kê nhanh số ca gian lận/mặc định phát hiện được
                        total_frauds = int(np.sum(batch_preds))
                        nn.metric("Số lượng giao dịch rủi ro phát hiện", f"{total_frauds} / {df_result.shape[0]}")
                        
                        nn.markdown("#### 📄 Bảng kết quả dự báo chi tiết:")
                        nn.dataframe(df_result, use_container_width=True)
                        
                        # Xuất file kết quả dự báo ra định dạng CSV để tải về
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_output = csv_buffer.getvalue()
                        
                        nn.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo (.CSV)",
                            data=csv_output,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv"
                        )
                else:
                    nn.error("❌ Định dạng tệp không hợp lệ hoặc dữ liệu trống.")
