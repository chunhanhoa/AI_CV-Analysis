# AI_CV-Analysis

## Giới thiệu

**AI_CV-Analysis** là một dự án nhằm áp dụng các kỹ thuật trí tuệ nhân tạo (AI) trong phân tích và xử lý hình ảnh, hỗ trợ các tác vụ như phân loại ảnh, nhận diện đối tượng và các ứng dụng thị giác máy tính khác.  
Dự án được xây dựng nhằm tối ưu hóa quá trình xử lý và phân tích dữ liệu hình ảnh với mục tiêu đem lại những kết quả chính xác và hiệu quả cao.

## Công nghệ sử dụng

- **Python:** Ngôn ngữ lập trình chính của dự án.
- **TensorFlow / PyTorch:** Thư viện học sâu để xây dựng và huấn luyện mô hình.
- **OpenCV:** Thư viện xử lý và phân tích hình ảnh.
- **NumPy:** Thư viện tính toán số học cho Python.
- **Matplotlib / Seaborn:** Thư viện trực quan hóa dữ liệu.

## Cài đặt

### Yêu cầu

- Python 3.8+ (khuyến nghị sử dụng Python 3.9 hoặc mới hơn)
- Pip (trình quản lý gói cho Python)
- Các thư viện và gói được liệt kê trong file `requirements.txt`

### Hướng dẫn cài đặt

1. **Clone repository:**

   ```bash
   git clone https://github.com/chunhanhoa/AI_CV-Analysis.git

2. **Tạo và kích hoạt môi trường ảo:**

    ```bash
    cd AI_CV-Analysis
    python -m venv venv
    # Trên Windows:
    venv\Scripts\activate
    # Trên macOS/Linux:
    source venv/bin/activate

3. **Cài đặt các thư viện cần thiết:**

   ```bash
   pip install -r requirements.txt

  ### Cấu trúc dự án
  
    data: Chứa dữ liệu đầu vào, bao gồm các hình ảnh dùng để phân tích.

    models: Chứa các mô hình đã được huấn luyện hoặc cấu hình sẵn để sử dụng.

    scripts: Chứa các script xử lý, huấn luyện và phân tích hình ảnh.
  
    utils: Chứa các hàm tiện ích hỗ trợ xử lý dữ liệu và hiển thị kết quả.

    notebooks: (Nếu có) Chứa các notebook Jupyter để thử nghiệm và trực quan hóa dữ liệu.    
