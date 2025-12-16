import json
import os
import re
import tempfile
from io import BytesIO
from typing import Dict, Any, Tuple

from fastapi import status, UploadFile
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from openai import OpenAI
from gtts import gTTS
from paddleocr import PaddleOCR
from PIL import Image

from app.core.llm import get_llm
from app.core.config import settings
from app.utils.exception_handler import CustomException
from app.schemas.sche_generate import (
    GenerateSentenceResponse,
    SpeechToTextResponse,
    OCRReadingResponse,
    ChatBotResponse,
)


# ==================== Helper Functions ====================

def cleanup_temp_file(file_path: str):
    """Hàm helper để xóa file tạm"""
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except:
            pass


# Khởi tạo PaddleOCR một lần để tái sử dụng (lazy loading)
_ocr_instance = None


def get_ocr_instance():
    """Lazy load PaddleOCR instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(use_angle_cls=True, lang='en')
    return _ocr_instance


def parse_llm_response(response_text: str, default_keys: list) -> Dict[str, Any]:
    """
    Parse response từ LLM thành dictionary
    
    Args:
        response_text: Text response từ LLM
        default_keys: List các keys mặc định nếu parse thất bại
    
    Returns:
        Dictionary chứa dữ liệu đã parse
    """
    response_text = response_text.strip()
    
    # Tìm JSON trong response (có thể có text thêm)
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Nếu không tìm thấy JSON, thử parse toàn bộ response
    try:
        return json.loads(response_text)
    except:
        # Fallback: tạo response mặc định
        default_data = {key: "Không thể phân tích" for key in default_keys}
        if "overall_feedback" in default_keys:
            default_data["overall_feedback"] = response_text[:500] if response_text else "Không có phản hồi"
        return default_data


# ==================== Service Functions ====================

def generate_sentence_service(vocabulary: str) -> GenerateSentenceResponse:
    """
    Service để sinh câu chứa từ vựng
    
    Args:
        vocabulary: Từ vựng cần sinh câu
    
    Returns:
        GenerateSentenceResponse chứa câu được sinh ra
    """
    vocabulary = vocabulary.strip()
    
    if not vocabulary:
        raise CustomException(
            http_code=status.HTTP_400_BAD_REQUEST,
            message="Từ vựng không được để trống"
        )
    
    # Tạo prompt để sinh câu chứa từ vựng
    prompt = f"""Hãy tạo một câu tiếng Anh ngắn gọn, dễ hiểu chứa từ vựng "{vocabulary}". 
Câu phải tự nhiên và giúp người học hiểu được cách sử dụng từ vựng này trong ngữ cảnh thực tế.
Chỉ trả về câu tiếng Anh, không cần giải thích thêm."""

    # Gọi LLM để sinh câu
    messages = [HumanMessage(content=prompt)]
    response = get_llm.invoke(messages)
    
    # Lấy câu được sinh ra từ response
    generated_sentence = response.content.strip()
    
    return GenerateSentenceResponse(
        sentence=generated_sentence,
        vocabulary=vocabulary
    )


async def speech_to_text_analysis_service(
    audio_content: bytes,
    audio_filename: str,
    topic: str = None
) -> SpeechToTextResponse:
    """
    Service để chuyển đổi speech to text và phân tích bài nói IELTS
    
    Args:
        audio_content: Nội dung file audio (bytes)
        audio_filename: Tên file audio
        topic: Đề bài/chủ đề bài nói (tùy chọn)
    
    Returns:
        SpeechToTextResponse chứa transcript và phân tích
    """
    # Khởi tạo OpenAI client
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    # Tạo file tạm để gửi cho Whisper
    file_ext = audio_filename.split('.')[-1] if audio_filename else 'mp3'
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(audio_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Gọi Whisper-1 để chuyển đổi speech to text
        with open(tmp_file_path, "rb") as audio_file_obj:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file_obj
            )
        
        transcript = transcript_response.text.strip()
        
        if not transcript:
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="Không thể chuyển đổi audio thành text. Vui lòng kiểm tra lại file audio."
            )
        
        # Tạo prompt để phân tích bài nói IELTS
        topic_info = f"\nĐề bài/Chủ đề: {topic}" if topic else ""
        prompt = f"""Bạn là một giám khảo IELTS chuyên nghiệp. Hãy phân tích bài nói sau đây của thí sinh và đưa ra đánh giá chi tiết.

{topic_info}

Nội dung bài nói của thí sinh:
"{transcript}"

Hãy phân tích và trả về kết quả theo định dạng JSON sau (chỉ trả về JSON, không có text thêm):
{{
    "band_score": "X.X (ví dụ: 6.5, 7.0, etc.)",
    "strengths": "Liệt kê các điểm mạnh của bài nói (ví dụ: từ vựng đa dạng, ngữ pháp chính xác...)",
    "weaknesses": "Liệt kê các điểm yếu cần cải thiện (ví dụ: thiếu từ nối, lỗi ngữ pháp ...)",
    "overall_feedback": "Nhận xét tổng quan về bài nói và đưa ra lời khuyên cụ thể để cải thiện"
}}

Lưu ý:
- Đánh giá dựa trên 3 tiêu chí IELTS Speaking: Lexical Resource, Grammatical Range and Accuracy, Pronunciation
- Band score từ 0.0 đến 9.0
- Đưa ra nhận xét chi tiết và cụ thể, có ví dụ minh họa nếu có thể"""
        
        # Gọi LLM để phân tích
        messages = [HumanMessage(content=prompt)]
        llm_response = get_llm.invoke(messages)
        
        # Parse response từ LLM
        analysis_data = parse_llm_response(
            llm_response.content,
            ["band_score", "strengths", "weaknesses", "overall_feedback"]
        )
        
        return SpeechToTextResponse(
            transcript=transcript,
            band_score=analysis_data.get("band_score", "N/A"),
            strengths=analysis_data.get("strengths", ""),
            weaknesses=analysis_data.get("weaknesses", ""),
            overall_feedback=analysis_data.get("overall_feedback", "")
        )
    
    finally:
        # Xóa file tạm
        cleanup_temp_file(tmp_file_path)


def text_to_speech_service(text: str, lang: str) -> Tuple[str, FileResponse]:
    """
    Service để chuyển đổi text to speech
    
    Args:
        text: Text cần chuyển đổi
        lang: Ngôn ngữ (mặc định: en)
    
    Returns:
        Tuple (file_path, FileResponse) - file_path để cleanup sau
    """
    text = text.strip()
    lang = lang.strip() if lang else "en"
    
    if not text:
        raise CustomException(
            http_code=status.HTTP_400_BAD_REQUEST,
            message="Text không được để trống"
        )
    
    # Tạo file tạm để lưu audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file_path = tmp_file.name
    
    try:
        # Sử dụng gTTS để tạo audio
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(tmp_file_path)
        
        # Kiểm tra file đã được tạo
        if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
            raise CustomException(
                http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                message="Không thể tạo file audio"
            )
        
        # Tạo FileResponse
        file_response = FileResponse(
            path=tmp_file_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
        
        return tmp_file_path, file_response
    
    except Exception as e:
        # Xóa file tạm nếu có lỗi
        cleanup_temp_file(tmp_file_path)
        raise


async def ocr_reading_analysis_service(
    image_content: bytes,
    image_filename: str,
    topic: str = None
) -> OCRReadingResponse:
    """
    Service để OCR ảnh thành text và phân tích bài viết Reading IELTS
    
    Args:
        image_content: Nội dung file ảnh (bytes)
        image_filename: Tên file ảnh
        topic: Đề bài/chủ đề bài viết (tùy chọn)
    
    Returns:
        OCRReadingResponse chứa extracted text và phân tích
    """
    # Kiểm tra file có dữ liệu không
    if not image_content or len(image_content) == 0:
        raise CustomException(
            http_code=status.HTTP_400_BAD_REQUEST,
            message="File ảnh không hợp lệ hoặc rỗng"
        )
    
    # Mở ảnh bằng PIL để kiểm tra
    try:
        image = Image.open(BytesIO(image_content))
        image.verify()  # Verify that it's a valid image
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_400_BAD_REQUEST,
            message=f"File ảnh không hợp lệ: {str(e)}"
        )
    
    # Tạo file tạm để lưu ảnh
    file_ext = image_filename.split('.')[-1] if image_filename else 'jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(image_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Sử dụng PaddleOCR để extract text
        ocr = get_ocr_instance()
        result = ocr.ocr(tmp_file_path)
        
        # Xử lý kết quả OCR
        extracted_text = ""
        if result and result[0]:
            # Kết hợp tất cả text từ các dòng
            text_lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_lines.append(line[1][0])  # line[1][0] là text được nhận diện
            extracted_text = "\n".join(text_lines).strip()
        
        if not extracted_text:
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="Không thể trích xuất text từ ảnh. Vui lòng kiểm tra lại chất lượng ảnh."
            )
        
        # Tạo prompt để phân tích bài viết Reading IELTS
        topic_info = f"\nĐề bài/Chủ đề: {topic}" if topic else ""
        prompt = f"""Bạn là một giám khảo IELTS chuyên nghiệp. Hãy phân tích bài viết Reading sau đây của thí sinh và đưa ra đánh giá chi tiết.

{topic_info}

Nội dung bài viết của thí sinh:
"{extracted_text}"

Hãy phân tích và trả về kết quả theo định dạng JSON sau (chỉ trả về JSON, không có text thêm):
{{
    "band_score": "X.X (ví dụ: 6.5, 7.0, etc.)",
    "strengths": "Liệt kê các điểm mạnh của bài viết (ví dụ: từ vựng đa dạng, ngữ pháp chính xác, cấu trúc rõ ràng...)",
    "weaknesses": "Liệt kê các điểm yếu cần cải thiện (ví dụ: lỗi chính tả, ngữ pháp, thiếu từ nối...)",
    "overall_feedback": "Nhận xét tổng quan về bài viết và đưa ra lời khuyên cụ thể để cải thiện",
    "suggestions": "Gợi ý cụ thể để cải thiện bài viết (ví dụ: sử dụng từ vựng phong phú hơn, cải thiện cấu trúc câu...)"
}}

Lưu ý:
- Đánh giá dựa trên các tiêu chí IELTS Writing/Reading: Task Achievement, Coherence and Cohesion, Lexical Resource, Grammatical Range and Accuracy
- Band score từ 0.0 đến 9.0
- Đưa ra nhận xét chi tiết và cụ thể, có ví dụ minh họa nếu có thể
- Phân tích cả về nội dung, cấu trúc, từ vựng và ngữ pháp"""
        
        # Gọi LLM để phân tích
        messages = [HumanMessage(content=prompt)]
        llm_response = get_llm.invoke(messages)
        
        # Parse response từ LLM
        analysis_data = parse_llm_response(
            llm_response.content,
            ["band_score", "strengths", "weaknesses", "overall_feedback", "suggestions"]
        )
        
        return OCRReadingResponse(
            extracted_text=extracted_text,
            band_score=analysis_data.get("band_score", "N/A"),
            strengths=analysis_data.get("strengths", ""),
            weaknesses=analysis_data.get("weaknesses", ""),
            overall_feedback=analysis_data.get("overall_feedback", ""),
            suggestions=analysis_data.get("suggestions", "")
        )
    
    finally:
        # Xóa file tạm
        cleanup_temp_file(tmp_file_path)


def chat_bot_service(
    question: str,
) -> ChatBotResponse:
    """Service chatbot giải đáp thắc mắc về tiếng Anh / IELTS cho học viên."""
    q = (question or "").strip()
    if not q:
        raise CustomException(
            http_code=status.HTTP_400_BAD_REQUEST,
            message="Câu hỏi không được để trống",
        )

    prompt = f"""
Bạn là một giáo viên tiếng Anh kiêm chuyên gia IELTS thân thiện và dễ hiểu.
Nhiệm vụ của bạn là giải đáp thắc mắc cho học viên về ngữ pháp, từ vựng, phát âm,
chiến lược làm bài thi IELTS, hoặc bất kỳ câu hỏi nào liên quan đến việc học tiếng Anh.

Câu hỏi/thắc mắc của học viên:
"{q}"

Yêu cầu về cách trả lời:
- Trả lời CHỦ YẾU bằng tiếng Việt, nhưng hãy đưa ví dụ minh hoạ bằng tiếng Anh khi cần.
- Giải thích rõ ràng, dễ hiểu cho người học Việt Nam.
- Giải thích ngắn gọn nhưng đầy đủ ý, có thể chia ý bằng bullet nếu cần.
- Ưu tiên cho ví dụ cụ thể, dễ áp dụng.

Chỉ trả về NỘI DUNG CÂU TRẢ LỜI cho học viên, không thêm JSON, không thêm tiền tố như 'Answer:'.
"""

    messages = [HumanMessage(content=prompt)]
    response = get_llm.invoke(messages)
    answer = (response.content or "").strip()

    if not answer:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Không thể tạo được câu trả lời từ mô hình. Vui lòng thử lại sau.",
        )

    return ChatBotResponse(answer=answer)
