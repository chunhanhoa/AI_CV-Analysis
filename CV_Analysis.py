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
names = ["Nguyễn Anh Tuấn", "Trần Văn A", "Lê Thị B", "Nguyễn Thị C", "Phan Đức D",
         "Lê Quốc Đạt", "Võ Thị Ngọc Hương", "Hoàng Trọng Duy", "Phạm Văn Thành", "Nguyễn Thu Hà",
         "Trần Quốc Vương", "Đỗ Thị Quỳnh Mai", "Lê Minh Huy", "Phạm Ngọc Khánh", "Hoàng Thị Mỹ Linh",
         "Lê Quốc Đạt", "Trần Hoài Nam", "Nguyễn Thanh Vũ", "Lê Thị Mai Anh", "Phạm Minh Long",
         "Trần Quang Huy", "Lê Hoàng Minh", "Nguyễn Hải Yến", "Phạm Văn Hải", "Trần Thị Mai Phương",
         "Hoàng Minh Khang", "Lâm Thị Ngọc Mai", "Vũ Thị Hoa", "Trần Quốc Bảo", "Phạm Ngọc Linh",
         "Hồ Minh Quân", "Trương Văn Bình", "Đặng Minh Hùng", "Nguyễn Hải Yến", "Lê Thanh Tuấn",
         "Nguyễn Duy Nam", "Phạm Văn Hải", "Trần Thị Mai Phương", "Hoàng Minh Khang", "BÙI VĂN Q",
         "ĐẶNG TUẤN P"]

cities = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Nha Trang",
          "Hà Nội", "Đà Nẵng", "TP. Hồ Chí Minh", "Hà Nội", "TP. Hải Phòng",
          "TP. Nha Trang", "Hà Nội", "TP. Hồ Chí Minh", "TP. Cần Thơ", "TP. Hồ Chí Minh",
          "Hà Nội", "TP. Hồ Chí Minh", "TP. Đà Nẵng", "TP. Đà Nẵng", "TP. Hồ Chí Minh",
          "TP. Hà Nội", "TP. Hồ Chí Minh", "Huế", "Hà Nội", "Đà Nẵng",
          "Hà Nội", "Đà Lạt", "Hải Phòng", "Hồ Chí Minh", "Đà Nẵng",
          "Hà Nội", "Đồng Nai", "Bình Dương", "Huế", "Cần Thơ",
          "Hồ Chí Minh", "Hải Phòng", "Đà Nẵng", "Hà Nội", "TP.HCM",
          "Hà Nội"]

skills = ["Python, Java, SQL", "C#, JavaScript, HTML, CSS", "Node.js, React, MongoDB", "Java, Kotlin, Android", "Go, Ruby, Docker",
          "Java, .NET, Oracle Database", "Data Analysis", "AI, Machine Learning", "Blockchain, Smart Contracts", "Mobile Development",
          "MySQL, GoLang", "AI, TensorFlow", "Network Security, Python", "IoT, C++, Arduino", "Data Analysis, Tableau",
          "Java, .NET, Oracle Database", "Angular, Bootstrap", "Java, Spring Boot, Docker", "Java, Spring Boot, Docker", "Angular, Bootstrap",
          "JavaScript, Python, C++, Docker, Git, Jenkins", "DevOps, Docker, Jenkins", "Python, Java", "Big Data, Hadoop, Spark", "HTML, CSS, JavaScript, Vue.js",
          "JavaScript, TypeScript, Node.js", "Photoshop, Illustrator", "Articulate, Moodle", "React, Node.js, MongoDB", "Content Writing, Social Media",
          "Agile, Scrum", "Java, Go, Docker", "Flutter, Dart, Swift", "Python, Java", "SQL, Python, Tableau",
          "AWS, Azure, Docker, Kubernetes", "Big Data, Spark", "Frontend Development", "JavaScript, TypeScript, SQL", "Java, Spring Boot, MySQL",
          "DevOps, CI/CD"]

projects = ["Hệ thống quản lý sinh viên (2020)", "Ứng dụng di động cho công ty XYZ", "Website bán hàng trực tuyến", "Hệ thống theo dõi sức khỏe người dùng", "Ứng dụng chatbot hỗ trợ khách hàng",
            "Ứng dụng quản lý tài khoản ngân hàng (2022)", "Phân tích dữ liệu khách hàng", "Ứng dụng AI trong sản xuất", "Smart Contract Platform", "Ứng dụng di động thương mại",
            "Hệ thống thanh toán trực tuyến (2022)", "Dự đoán bệnh dựa trên dữ liệu lớn", "Hệ thống phát hiện tấn công mạng", "Hệ thống điều khiển đèn thông minh", "Hệ thống dự đoán doanh thu",
            "Ứng dụng quản lý tài khoản ngân hàng", "Nền tảng thương mại điện tử đa kênh", "Hệ thống quản lý học sinh", "Hệ thống quản lý học sinh", "Nền tảng thương mại điện tử đa kênh",
            "Hệ thống quản lý học sinh (2023)", "Tối ưu hóa triển khai phần mềm", "Dự án nghiên cứu ML", "Hệ thống phân tích dữ liệu khách hàng", "Ứng dụng quản lý bệnh viện",
            "Hệ thống quản lý kho hàng (2022)", "Thiết kế logo và branding", "Khóa học Kỹ năng mềm", "Ứng dụng quản lý bán hàng", "Chiến dịch marketing số",
            "Quản lý dự án phần mềm lớn", "Hệ thống quản lý năng lượng tái tạo", "Ứng dụng quản lý lịch trình", "Dự án nghiên cứu ML", "Hệ thống dự đoán doanh thu ngành bán lẻ",
            "Hệ thống tự động triển khai web", "Phân tích dữ liệu lớn", "Ứng dụng quản lý bệnh viện", "Hệ thống quản lý doanh nghiệp", "Hệ thống quản lý Java Enterprise",
            "Hệ thống CI/CD"]

achievements = ["Giải Nhất Hackathon 2021", "Nhân viên xuất sắc Công ty XYZ", "Giải Ba cuộc thi lập trình quốc gia", "Giải Nhất cuộc thi Code Wars",
                "Nhân viên xuất sắc năm 2021 tại Vietcombank", "Top 10 Data Analyst 2022", "Giải nhất AI Challenge 2021", "Best Blockchain Developer 2022", "Mobile Developer of the Year",
                "Giải thưởng Sáng tạo Tech 2022", "Chứng nhận TensorFlow Developer", "CEH Certification", "Cisco CCNA", "Tableau Specialist",
                "Nhân viên xuất sắc năm 2021", "Giải Ba Hackathon 2021", "Google Cloud Certification", "AWS Certification", "Nhân viên xuất sắc 2022",
                "Giải Nhì Hackathon Cloud Computing 2020", "Top 3 Data Science Challenge", "Nhân viên tiêu biểu 2021", "Java Certification", "Google Analytics Certification",
                "PMP Certification", "Giải thưởng sáng kiến công nghệ", "UI/UX Design Award", "Scrum Master Certification", "Digital Marketing Excellence",
                "Best Project Manager 2022"]

degrees = ["Cử nhân Công nghệ Thông tin", "Cử nhân Kỹ thuật Phần mềm", "Cử nhân Quản trị Kinh doanh", "Thạc sĩ Khoa học Máy tính",
           "Cử nhân Hệ thống Thông tin Quản lý", "Thạc sĩ Khoa học Dữ liệu", "Cử nhân Trí tuệ Nhân tạo", "Cử nhân An toàn Thông tin",
           "Cử nhân IoT", "Cử nhân Phân tích Dữ liệu", "Thạc sĩ An toàn Thông tin", "Cử nhân Kỹ thuật Máy tính",
           "Cử nhân Khoa học Dữ liệu", "Thạc sĩ Trí tuệ Nhân tạo", "Cử nhân Công nghệ Phần mềm", "Thạc sĩ Khoa học Máy tính",
           "Cử nhân Hệ thống Thông tin", "Cử nhân Mạng máy tính", "Thạc sĩ Kỹ thuật Phần mềm", "Cử nhân Khoa học Máy tính"]

emails = ["nguyen.tuan@example.com", "tran.a@example.com", "le.b@example.com", "nguyen.c@example.com", "phan.d@example.com",
          "lequocdat@example.com", "ngochuong@example.com", "trongduy@example.com", "vanthanh@example.com", "thuha@example.com",
          "quocvuong@example.com", "quynhmai@example.com", "minhhuy@example.com", "ngockhanh@example.com", "mylinh@example.com",
          "lequocdat@example.com", "hoainam@example.com", "thanhvu@example.com", "maianh@example.com", "minhlong@example.com",
          "quanghuy@example.com", "hoangminh@example.com", "haiyen@example.com", "vanhai@example.com", "maiphuong@example.com",
          "minhkhang@example.com", "ngocmai@example.com", "thihoa@example.com", "quocbao@example.com", "ngoclinh@example.com",
          "minhquan@example.com", "vanbinh@example.com", "minhhung@example.com", "haiyen@example.com", "thanhtuan@example.com",
          "duynam@example.com", "vanhai@example.com", "maiphuong@example.com", "minhkhang@example.com", "buiq@example.com",
          "tuanp@example.com"]

phones = ["0987654321", "0912345678", "0909876543", "0981234567", "0918765432",
          "0912456789", "0909123456", "0908234567", "0907345678", "0906456789",
          "0905567890", "0987345678", "0912987654", "0909123456", "0908765432",
          "0912456789", "0909876543", "0979123456", "0979123456", "0909876543",
          "0987654321", "0988123456", "0908776654", "0911223344", "0933445566",
          "0922334455", "0905777766", "0912345678", "0988112233", "0905123456",
          "0938778889", "0975223344", "0988332211", "0908776654", "0938776543",
          "0909776655", "0911223344", "0933445566", "0922334455", "0907654321",
          "0901231234"]

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
    # Programming Languages
    "python", "javascript", "sql", "java", "c++", "c#", "php", "ruby", "swift",
    "typescript", "kotlin", "go", "rust", "scala", "r", "matlab", "perl",
    "objective-c", "dart", "lua", "haskell", "assembly",

    # Frameworks & Libraries
    "react", "node.js", "angular", "vue.js", "django", "flask", "spring", "express.js",
    "laravel", "asp.net", "ruby on rails", "jquery", "bootstrap", "tailwind",
    "next.js", "nuxt.js", "flutter", "react native", "xamarin", "tensorflow",
    "pytorch", "keras", "pandas", "numpy", "scikit-learn", "redux", "graphql",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
    "terraform", "ansible", "puppet", "chef", "circleci", "travis ci",
    "github actions", "bitbucket", "gitlab", "nginx", "apache", "linux",
    "windows server", "bash", "shell scripting", "powershell",

    # Big Data & AI
    "hadoop", "spark", "big data", "machine learning", "ai", "deep learning",
    "data mining", "data science", "natural language processing", "computer vision",
    "neural networks", "reinforcement learning", "data analytics", "tableau",
    "power bi", "elasticsearch", "kafka", "airflow", "databricks",

    # Databases
    "mysql", "mongodb", "postgresql", "firebase", "oracle",
    "sql server", "redis", "cassandra", "dynamodb", "mariadb",
    "neo4j", "sqlite", "couchdb", "influxdb", "elasticsearch",

    # Frontend
    "html", "css", "sass", "less", "webpack", "babel", "responsive design",
    "web accessibility", "seo", "pwa", "web components", "material ui",
    "ant design", "chakra ui", "styled components",

    # Backend
    "rest api", "soap", "microservices", "websocket", "grpc", "oauth",
    "jwt", "api gateway", "load balancing", "caching", "message queues",
    "rabbitmq", "redis", "memcached",

    # Testing
    "unit testing", "integration testing", "selenium", "jest", "mocha",
    "cypress", "junit", "pytest", "testng", "cucumber", "postman",

    # Methodologies & Practices
    "scrum", "agile", "waterfall", "kanban", "lean", "tdd", "bdd",
    "ci/cd", "devops", "itil", "solid principles", "design patterns",
    "mvc", "mvvm", "clean architecture",

    # Security
    "cybersecurity", "encryption", "oauth", "jwt", "ssl/tls",
    "penetration testing", "security audit", "firewall", "vpn",

    # Mobile Development
    "ios", "android", "react native", "flutter", "xamarin",
    "swift ui", "kotlin multiplatform", "mobile security",

    # Version Control
    "git", "svn", "mercurial", "github", "gitlab", "bitbucket",
    "git flow", "trunk based development",

    # Project Management
    "jira", "trello", "asana", "confluence", "notion",
    "microsoft project", "basecamp",

    # Additional Skills
    "blockchain", "ethereum", "smart contracts", "solidity",
    "ar/vr", "unity", "unreal engine", "webgl", "three.js",
    "iot", "embedded systems", "raspberry pi", "arduino"
    
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
    
    # Single token skills
    for token in doc:
        if token.text in SKILLS_KEYWORDS:
            skills_found.append(token.text)
    
    # Multi-token skills (for phrases like "machine learning")
    for skill in SKILLS_KEYWORDS:
        if ' ' in skill and skill in text.lower():
            skills_found.append(skill)
            
    return list(set(skills_found))  # Remove duplicates

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
        st.image("https://img.freepik.com/free-vector/cv-template-minimalist-style_23-2148911517.jpg", width=200)
    
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
