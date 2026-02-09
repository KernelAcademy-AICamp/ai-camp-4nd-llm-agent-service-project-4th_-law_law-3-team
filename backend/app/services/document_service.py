import os
from jinja2 import Environment, FileSystemLoader

class DocumentService:
    def __init__(self) -> None:
        # 템플릿 디렉토리 설정 (app/templates)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, "../templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def generate_demand_letter(
        self,
        data: dict,
        format: str = "text",
        output_path: str | None = None
    ) -> str:
        """내용증명 생성"""
        # 필수 필드 검증
        required_fields = ["recipient", "sender", "content"]
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"필수 데이터 누락: {', '.join(missing)}")

        if format == "text":
            template = self.env.get_template("demand_letter.j2")
            return template.render(**data)
        
        elif format == "pdf":
            if not output_path:
                raise ValueError("PDF 생성 시 output_path는 필수입니다.")
                
            from reportlab.pdfgen import canvas
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.pagesizes import A4
            
            # 한글 폰트 설정 (맑은 고딕 등 시스템 폰트 활용)
            # 여기서는 편의상 기본 폰트나 시스템 폰트 경로를 지정해야 함
            # 윈도우 환경이므로 Malgun Gothic 시도
            try:
                font_path = "C:/Windows/Fonts/malgun.ttf"
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont("Malgun", font_path))
                    font_name = "Malgun"
                else:
                    font_name = "Helvetica" # 한글 깨짐 주의 (폴백)
            except Exception:
                font_name = "Helvetica"

            c = canvas.Canvas(output_path, pagesize=A4)
            c.setFont(font_name, 12)
            
            # 텍스트 렌더링 (줄바꿈 처리 필요)
            text = self.env.get_template("demand_letter.j2").render(**data)
            y = 800
            for line in text.split('\n'):
                c.drawString(50, y, line)
                y -= 20
                
            c.save()
            return output_path
            
        else:
            raise ValueError(f"지원하지 않는 포맷입니다: {format}")

    def generate_pdf_from_text(self, text: str, output_path: str) -> str:
        """텍스트 내용을 PDF로 생성"""
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.pagesizes import A4
        
        # 폰트 설정
        font_name = "Helvetica"
        try:
            # 윈도우/리눅스 폰트 경로 확인
            font_paths = [
                "C:/Windows/Fonts/malgun.ttf", # Windows
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", # Linux (Nanum)
                "/System/Library/Fonts/AppleSDGothicNeo.ttc", # Mac
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    pdfmetrics.registerFont(TTFont("Malgun", path))
                    font_name = "Malgun"
                    break
        except Exception as e:
            print(f"폰트 로드 실패: {e}")

        c = canvas.Canvas(output_path, pagesize=A4)
        c.setFont(font_name, 11)
        
        # 텍스트 그리기 (간단한 줄바꿈 처리)
        y = 800
        line_height = 20
        margin_left = 50
        max_width = 500
        
        for line in text.split('\n'):
            # 긴 줄 처리 (지나치게 길면 자르거나 줄바꿈.. 여기선 단순화)
            # ReportLab의 simpledoctemplate을 쓰면 좋지만, 여기선 캔버스로 빠르게 구현
            
            # 페이지 넘김 처리
            if y < 50:
                c.showPage()
                c.setFont(font_name, 11)
                y = 800
                
            c.drawString(margin_left, y, line)
            y -= line_height
            
        c.save()
        return output_path

    def generate_hwpx_from_text(self, text: str, output_path: str) -> str:
        """텍스트 내용을 HWPX(한글) 파일로 생성"""
        from hwpx.document import HwpxDocument
        
        # 빈 문서 생성 (HwpxDocument.new() 사용)
        doc = HwpxDocument.new()
        
        # 텍스트를 문단별로 추가
        for line in text.split('\n'):
            doc.add_paragraph(line)
        
        # 파일 저장
        doc.save(output_path)
        return output_path

    def generate_docx_from_text(self, text: str, output_path: str) -> str:
        """텍스트 내용을 DOCX(워드) 파일로 생성 - 한글에서도 열 수 있음"""
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        doc = Document()
        
        # 텍스트를 문단별로 추가
        for line in text.split('\n'):
            para = doc.add_paragraph(line)
            # 기본 폰트 설정
            for run in para.runs:
                run.font.name = '맑은 고딕'
                run.font.size = Pt(11)
        
        # 파일 저장
        doc.save(output_path)
        return output_path
