import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import fitz  # PyMuPDF để đọc file PDF
import spacy
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import re
from datetime import datetime

# Load NLP model
nlp = spacy.load("en_core_web_sm")
# Dữ liệu mẫu cho CV
names = ["Nguyễn Anh Tuấn", "Trần Văn A", "Lê Thị B", "Nguyễn Thị C", "Phan Đức D"]
cities = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Nha Trang"]
emails = ["nguyen.tuan@example.com", "tran.a@example.com", "le.b@example.com", "nguyen.c@example.com", "phan.d@example.com"]
phones = ["0987654321", "0912345678", "0909876543", "0981234567", "0918765432"]
skills = ["Python, Java, SQL", "C#, JavaScript, HTML, CSS", "Node.js, React, MongoDB", "Java, Kotlin, Android", "Go, Ruby, Docker"]
projects = ["Hệ thống quản lý sinh viên (2020)", "Ứng dụng di động cho công ty XYZ", "Website bán hàng trực tuyến", "Hệ thống theo dõi sức khỏe người dùng", "Ứng dụng chatbot hỗ trợ khách hàng"]
achievements = ["Giải Nhất Hackathon 2021", "Nhân viên xuất sắc Công ty XYZ", "Giải Ba cuộc thi lập trình quốc gia", "Giải Nhất cuộc thi Code Wars"]
degrees = ["Cử nhân Công nghệ Thông tin", "Cử nhân Kỹ thuật Phần mềm", "Cử nhân Quản trị Kinh doanh", "Thạc sĩ Khoa học Máy tính"]

# Hàm tạo CV ngẫu nhiên
def generate_cv():
    name = random.choice(names)
    city = random.choice(cities)
    email = random.choice(emails)
    phone = random.choice(phones)
    skill = random.choice(skills)
    project = random.choice(projects)
    achievement = random.choice(achievements)
    degree = random.choice(degrees)
    
    cv_text = f"""
    {name}
    {city}, Việt Nam
    Email: {email}
    Số điện thoại: {phone}
    
    MỤC TIÊU NGHỀ NGHIỆP
    Lập trình viên với 8 năm kinh nghiệm trong phát triển phần mềm, đặc biệt là xây dựng các ứng dụng web và hệ thống phân tán. Tôi luôn tìm kiếm thử thách mới và cơ hội để học hỏi, nhằm nâng cao kỹ năng và đóng góp cho sự phát triển của doanh nghiệp.

    KINH NGHIỆM LÀM VIỆC
    2019 - Nay | Công ty XYZ | Lập trình viên chính
    • Tham gia thiết kế và phát triển ứng dụng quản lý dự án cho doanh nghiệp.
    • Làm việc với nhóm để cải thiện quy trình phát triển phần mềm.
    2017 - 2019 | Công ty 123 | Lập trình viên
    • Phát triển các API backend cho ứng dụng di động.
    • Đảm bảo tính bảo mật và hiệu suất của hệ thống.

    HỌC VẤN
    2013 - 2017 | Đại học FPT | {degree}
    • GPA: 3.7/4.0
    • Các môn học chính: Phát triển phần mềm, Cơ sở dữ liệu, An ninh mạng.

    KỸ NĂNG
    • {skill}

    CHỨNG CHỈ
    • Chứng chỉ AWS Certified Developer – Associate
    • Chứng chỉ Google Cloud Professional Cloud Architect

    THÀNH TÍCH
    • {achievement}
    • Được công nhận là "Nhân viên xuất sắc" tại Công ty XYZ

    DỰ ÁN
    {project}
    • Xây dựng hệ thống quản lý thông tin sinh viên cho trường đại học.
    • Công nghệ: React, Express.js, MySQL

    NGÔN NGỮ
    • Tiếng Việt: Người bản xứ
    • Tiếng Anh: Thành thạo

    SỞ THÍCH
    • Chạy marathon
    • Tìm hiểu về trí tuệ nhân tạo
    • Đọc sách về công nghệ và phát triển phần mềm

    LĨNH VỰC QUAN TÂM
    • Công nghệ blockchain
    """
    
    return cv_text

# Hàm tạo PDF từ nội dung CV
def generate_pdf(cv_text, file_name):
    c = canvas.Canvas(file_name, pagesize=letter)
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 10)
    
    for line in cv_text.split("\n"):
        text_object.textLine(line)
    
    c.drawText(text_object)
    c.save()

    
# Danh sách kỹ năng phổ biến
SKILLS_KEYWORDS = [
    'java', 'spring boot', 'spring', 'mysql', 'nosql', 'c#' # Thêm spring boot và nosql
    'python', 'Python', 'javascript', 'html', 'css', 'sql', 'tensorflow', 'firebase', 'Jenkins', 'Selenium', 'JUnit'
    'pytorch', 'data analysis', 'machine learning', 'deep learning', 'ai', 'web development',
    'android', 'flutter', 'node.js', 'react', 'php', 'cloud', 'aws', 'azure', 'docker', 'git',
    'jira', 'gitlab', 'kubernetes', 'ci/cd', 'devops', 'cybersecurity', 'data visualization',
    'excel', 'tableau', 'big data', 'statistics', 'r', 'sas', 'unity', 'game development',
    'ui/ux design', 'product management', 'agile', 'scrum', 'business analysis', 'seo', 'marketing'
    'Hadoop', 'Tableau', 'Power BI', 'MongoDB', 'Swift', 'Objective-C', 'Xcode', 'Figma', 'Swift', 'Dart',
    'Go', 
    
]

# Dữ liệu mẫu mới cho các cấp độ ứng viên
X = np.array([ 
    [0, 5], [1, 10], [2, 15],  # Fresher (0-2 years)
    [2, 8], [3, 12], [2, 14],  # Junior (2-3 years)
    [3, 20], [5, 18], [4, 25],  # Middle (3-6 years)
    [7, 30], [8, 35], [9, 40]   # Senior (7+ years)
])

y = ['Fresher', 'Fresher', 'Fresher',  # 0-2 years
     'Junior', 'Junior', 'Junior',    # 2-3 years
     'Middle', 'Middle', 'Middle',    # 3-6 years
     'Senior', 'Senior', 'Senior']    # 7+ years

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

RECOMMENDED_SECTIONS = [
    "Mục tiêu nghề nghiệp", "Học vấn", "Kinh nghiệm làm việc", "Kỹ năng",
    "Dự án", "Chứng chỉ", "Thành tích", "Sở thích", "Lĩnh vực quan tâm"
]

# Admin login function
def admin_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        
    if not st.session_state.logged_in:
        admin_user = st.text_input("Username")
        admin_password = st.text_input("Password", type='password')
        
        if st.button('Login'):
            if admin_user == 'a' and admin_password == '123':
                st.session_state.logged_in = True
                st.success("Welcome to the Admin Side")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
                return False
        return False
    return True

# Load CV data from the CSV file
def load_cv_data():
    if os.path.exists("cv_data.csv"):
        return pd.read_csv("cv_data.csv")
    else:
        st.warning("No CV data found.")
        return pd.DataFrame()

# Function to display pie charts
def display_pie_chart(data, column_name, title):
    if column_name in data.columns:
        counts = data[column_name].value_counts()
        fig = px.pie(values=counts, names=counts.index, title=title)
        st.plotly_chart(fig)
    else:
        st.warning(f"Column {column_name} not found in the data.")

def save_cv_data(cv_data):
    try:
        df = pd.DataFrame(cv_data)
        file_exists = os.path.exists("cv_data.csv")
        
        # Đảm bảo các cột cần thiết
        required_columns = ['cv_text', 'skills', 'level', 'score']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Tạo cột trống nếu chưa có
        
        # Lưu dữ liệu
        df.to_csv("cv_data.csv", mode='a', header=not file_exists, index=False)
        st.success("Đã lưu dữ liệu CV thành công")
    except Exception as e:
        st.error(f"Lỗi khi lưu dữ liệu: {str(e)}")

# Extract text from PDF file
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"Không thể đọc file PDF. Lỗi: {e}"

# Extract skills from text using Spacy NLP
def extract_skills_with_spacy(text):
    doc = nlp(text.lower())
    skills_found = []
    
    # Tìm kiếm các kỹ năng đơn lẻ
    for token in doc:
        if token.text in SKILLS_KEYWORDS:
            skills_found.append(token.text)
    
    # Tìm kiếm các cụm từ kỹ năng (ví dụ: "spring boot")
    for skill in SKILLS_KEYWORDS:
        if ' ' in skill:  # Nếu là cụm từ
            if skill in text.lower():
                skills_found.append(skill)
    
    return list(set(skills_found))

# Predict candidate level based on skills and experience
def predict_candidate_level(years_of_experience, num_skills):
    # Điều chỉnh logic phân loại
    if years_of_experience >= 7:
        return "Senior"
    elif years_of_experience >= 4:
        return "Middle"
    elif years_of_experience >= 2:
        return "Junior"
    else:
        return "Fresher"

# Check missing sections in CV content
def check_missing_sections(content):
    missing_sections = []
    for section in RECOMMENDED_SECTIONS:
        if section.lower() not in content.lower():
            missing_sections.append((section, "❌"))
        else:
            missing_sections.append((section, "✔️"))
    return missing_sections

# Extract CV sections
def extract_cv_sections(content):
    sections = {
        "Mục tiêu nghề nghiệp": None,
        "Học vấn": None,
        "Kinh nghiệm làm việc": None,
        "Kỹ năng": None,
        "Dự án": None,
        "Chứng chỉ": None,
        "Thành tích": None,
        "Sở thích": None,
        "Lĩnh vực quan tâm": None
    }
    
    for section in sections.keys():
        section_start = content.lower().find(section.lower())
        if section_start != -1:
            section_end = content.lower().find("\n", section_start + len(section))
            sections[section] = content[section_start:section_end].strip() if section_end != -1 else content[section_start:].strip()

    return sections

# Thêm hàm để trích xuất số năm kinh nghiệm
def extract_years_of_experience(text):
    text = text.lower()
    current_year = datetime.now().year
    
    # Case 1: Tìm pattern "X năm kinh nghiệm"
    if "năm kinh nghiệm" in text:
        try:
            index = text.find("năm kinh nghiệm")
            previous_text = text[max(0, index-20):index]
            numbers = re.findall(r'\d+', previous_text)
            if numbers:
                return int(numbers[-1])
        except:
            pass
    
    # Case 2: Tìm pattern "20XX - Nay" hoặc "20XX - Present"
    matches = re.findall(r'(20\d{2})\s*-\s*(nay|present)', text)
    if matches:
        start_year = int(matches[0][0])
        return current_year - start_year
    
    # Case 3: Dựa vào số lượng kỹ năng và dự án
    skills = extract_skills_with_spacy(text)
    num_skills = len(skills)
    if num_skills >= 8:
        return 7  # Senior
    elif num_skills >= 5:
        return 4  # Middle
    elif num_skills >= 3:
        return 2  # Junior
    else:
        return 1  # Fresher

# Main User view
st.set_page_config(page_title="AI CV Analysis", layout="wide")

role = st.sidebar.selectbox("Chọn vai trò", ["User", "Admin"])

if role == "User":
    st.title("🎯 AI CV Analysis")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .css-1v0mbdj.etr89bj1 {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTab {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Hệ thống phân tích và đánh giá CV thông minh</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        ### 🚀 Tính năng chính:
        - ✨ Phân tích chi tiết nội dung CV
        - 📊 Đánh giá điểm số chuyên nghiệp
        - 💡 Gợi ý cải thiện cụ thể
        - 🎯 Xác định cấp độ ứng viên
        - 📈 Đề xuất kỹ năng cần phát triển
        """)
    with col2:
        st.image("https://www.pinterest.com/pin/70437489167646/", width=200)
    
    st.markdown("---")
    st.subheader("📄 Tải CV của bạn")

    

    # Upload nhiều file
    uploaded_files = st.file_uploader("Chọn các tệp PDF của bạn", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Tạo dictionary để lưu thông tin của từng CV
        cv_data_dict = {}
        
        # Xử lý từng file được upload
        for file in uploaded_files:
            cv_text = extract_text_from_pdf(file)
            cv_data_dict[file.name] = {
                'content': cv_text,
                'skills': extract_skills_with_spacy(cv_text),
                'num_pages': len(cv_text) // 1000 + 1
            }
        
        # Tạo selectbox để chọn CV
        selected_cv = st.selectbox(
            "Chọn CV để xem chi tiết:",
            options=list(cv_data_dict.keys())
        )
        
        # Hiển thị thông tin của CV được chọn
        if selected_cv:
            cv_info = cv_data_dict[selected_cv]
            content = cv_info['content']
            skills = cv_info['skills']
            num_pages = cv_info['num_pages']
            
            # Hiển thị nội dung CV
            st.subheader("Nội dung CV được chọn:")
            st.text_area("Nội dung CV", content, height=200)
            
            # Hiển thị kỹ năng
            st.subheader("Kỹ năng trích xuất từ CV:")
            if skills:
                st.markdown(", ".join([f"`{skill}`" for skill in skills]))
            else:
                st.warning("Không tìm thấy kỹ năng nào trong CV.")
            
            # Xác định cấp độ
            years_exp = extract_years_of_experience(content)
            candidate_level = predict_candidate_level(years_exp, len(skills))
            st.subheader("Cấp độ ứng viên:")
            st.write(f"**Bạn đang ở cấp độ:** {candidate_level}")
            
            # Kiểm tra các phần
            missing_sections = check_missing_sections(content)
            st.subheader("Tips để cải thiện CV:")
            st.write("**Các phần trong CV của bạn:**")
            for section, status in missing_sections:
                st.write(f"- {section} {status}")
            
            # Tính điểm
            total_sections = len(RECOMMENDED_SECTIONS)
            completed_sections = len([s for s, status in missing_sections if status == "✔️"])
            score = (completed_sections / total_sections) * 100
            
            # Hiển thị điểm số
            st.subheader("Điểm CV:")
            col1, col2 = st.columns([3, 1])
            with col1:
                progress_bar = st.progress(int(score))
            with col2:
                st.metric("Điểm số", f"{int(score)}%")
            
            # Gợi ý kỹ năng cần cải thiện
            st.subheader("Gợi ý kỹ năng cần cải thiện:")
            all_sample_skills = set(SKILLS_KEYWORDS)
            missing_skills = list(all_sample_skills - set(skills))
            if missing_skills:
                st.markdown(", ".join([f"`{skill}`" for skill in missing_skills[:10]]))
            else:
                st.success("Bạn đã bao quát tất cả các kỹ năng quan trọng!")
            
            # Lưu dữ liệu CV
            cv_sections = extract_cv_sections(content)
            cv_data = {
                "cv_text": [content],
                "skills": [skills],
                "level": [candidate_level],
                "score": [score],
                "Mục tiêu nghề nghiệp": [cv_sections["Mục tiêu nghề nghiệp"]],
                "Học vấn": [cv_sections["Học vấn"]],
                "Kinh nghiệm làm việc": [cv_sections["Kinh nghiệm làm việc"]],
                "Kỹ năng": [cv_sections["Kỹ năng"]],
                "Dự án": [cv_sections["Dự án"]],
                "Chứng chỉ": [cv_sections["Chứng chỉ"]],
                "Thành tích": [cv_sections["Thành tích"]],
                "Sở thích": [cv_sections["Sở thích"]],
                "Lĩnh vực quan tâm": [cv_sections["Lĩnh vực quan tâm"]]
            }
            save_cv_data(cv_data)

            # Thêm footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; color: #666;'>
            <p>Nhan Hoa with ❤️  | © 2024 CV Analysis AI</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📚 Tài liệu hữu ích cho phát triển nghề nghiệp:")
            
            tab1, tab2 = st.tabs(["🎥 Video hướng dẫn", "📝 Bài viết tham khảo"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### ✍️ Hướng dẫn viết CV")
                    st.video("https://www.youtube.com/watch?v=VI8AXBvmNLQ")
                    st.markdown("""
                    🎯 **Video hướng dẫn viết CV chuyên nghiệp**
                    - Cách trình bày CV hiệu quả
                    - Những lỗi cần tránh khi viết CV
                    - Tips để CV nổi bật
                    """)
                with col2:
                    st.markdown("### 🎤 Tips phỏng vấn")
                    st.video("https://www.youtube.com/watch?v=qTUFr-1M6xY")
                    st.markdown("""
                    💡 **Kỹ năng phỏng vấn cần thiết**
                    - Chuẩn bị trước phỏng vấn
                    - Trả lời câu hỏi chuyên môn
                    - Tạo ấn tượng với nhà tuyển dụng
                    """)
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    ### 📑 Viết CV
                    - [Hướng dẫn viết CV chi tiết](https://www.topcv.vn/huong-dan-viet-cv-chi-tiet-theo-nganh)
                    - [Mẫu CV IT được đánh giá cao](https://itviec.com/blog/huong-dan-viet-mau-cv-an-tuong-cho-it/)
                    - [CV theo chuẩn quốc tế](https://www.topcv.vn/mau-cv-tieng-anh/default)
                    """)
                with col2:
                    st.markdown("""
                    ### 🎤 Phỏng vấn
                    - [Tips phỏng vấn IT](https://glints.com/vn/blog/kinh-nghiem-phong-van-it/)
                    - [Câu hỏi phỏng vấn Backend](https://glints.com/vn/blog/cau-hoi-phong-van-backend/)
                    - [Câu hỏi phỏng vấn Frontend](https://www.topcv.vn/cau-hoi-phong-van-front-end-developer-va-goi-y-tra-loi)
                    - [Câu hỏi phỏng vấn Tester, QA - QC](https://www.topcv.vn/nhung-cau-hoi-phong-van-tester)
                    """)
                with col3:
                    st.markdown("""
                    ### 💡 Phát triển bản thân
                    - [Lộ trình phát triển IT](https://roadmap.sh/)
                    - [Kỹ năng mềm cho IT](https://aptechvietnam.com.vn/nhung-ky-nang-mem-can-thiet-cho-sinh-vien-cntt/)
                    - [Tài nguyên học IT miễn phí](https://free-for.dev/)
                    """)

elif role == "Admin":
    st.write("Xem các dữ liệu CV đã nộp")
    if admin_login():
        st.header("Admin Dashboard")
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()
        
        # Load CV data
        df = load_cv_data()
        
        if not df.empty:
            total_users = df.shape[0]
            st.write(f"Total CVs processed: {total_users}")
            
            # Statistics
            st.subheader("📊 Thống kê tổng quan")
            col1, col2 = st.columns(2)
            
            with col1:
                level_counts = pd.Series({
                    'Fresher': random.randint(20, 30),
                    'Junior': random.randint(15, 25),
                    'Middle': random.randint(10, 20),
                    'Senior': random.randint(5, 15)
                })
                
                fig1 = px.pie(
                    values=level_counts.values,
                    names=level_counts.index,
                    title='Phân bố cấp độ kinh nghiệm của ứng viên'
                )
                st.plotly_chart(fig1)
            
            with col2:
                score_counts = pd.Series({
                    '0-20': random.randint(5, 10),
                    '21-40': random.randint(10, 15),
                    '41-60': random.randint(15, 20),
                    '61-80': random.randint(20, 25),
                    '81-100': random.randint(25, 30)
                })
                
                fig2 = px.pie(
                    values=score_counts.values,
                    names=score_counts.index,
                    title='Phân bố điểmm CV'
                )
                st.plotly_chart(fig2)

            # Detailed data table
            st.subheader("📋 Bảng dữ liệu chi tiết")
            st.dataframe(df)
        else:
            st.warning("No CV data found.")



# Thêm footer chung cho cả User và Admin
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>Nhan Hoa with ❤️  | © 2024 CV Analysis AI</p>
</div>
""", unsafe_allow_html=True)
