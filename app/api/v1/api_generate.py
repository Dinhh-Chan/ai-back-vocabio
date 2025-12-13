from typing import Any
from fastapi import APIRouter, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse

from app.utils.exception_handler import CustomException
from app.schemas.sche_response import DataResponse
from app.schemas.sche_generate import (
    GenerateSentenceRequest,
    GenerateSentenceResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    OCRReadingResponse
)
from app.services.srv_generate import (
    generate_sentence_service,
    speech_to_text_analysis_service,
    text_to_speech_service,
    ocr_reading_analysis_service,
    cleanup_temp_file
)

router = APIRouter(prefix=f"/generate")


@router.post(
    "/generate-sentence",
    response_model=DataResponse[GenerateSentenceResponse],
    status_code=status.HTTP_200_OK,
)
def generate_sentence(request: GenerateSentenceRequest) -> Any:
    """
    API sinh câu chứa từ vựng
    
    - Nhận từ vựng từ người dùng
    - Sử dụng LLM để sinh câu chứa từ vựng đó
    """
    try:
        result = generate_sentence_service(request.vocabulary)
        
        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=result,
            message="Sinh câu thành công"
        )
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi sinh câu: {str(e)}"
        )


@router.post(
    "/speech-to-text-analysis",
    response_model=DataResponse[SpeechToTextResponse],
    status_code=status.HTTP_200_OK,
)
async def speech_to_text_analysis(
    audio_file: UploadFile = File(..., description="File audio bài nói IELTS"),
    topic: str = Form(None, description="Đề bài/chủ đề bài nói (tùy chọn)")
) -> Any:
    """
    API chuyển đổi speech to text và phân tích bài nói IELTS
    
    - Nhận file audio từ người dùng
    - Sử dụng Whisper-1 để chuyển đổi thành text
    - Gửi text cho LLM để phân tích band điểm, điểm mạnh, điểm yếu
    """
    try:
        # Kiểm tra file audio
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="File phải là định dạng audio"
            )
        
        # Đọc file audio
        audio_content = await audio_file.read()
        
        # Gọi service để xử lý
        result = await speech_to_text_analysis_service(
            audio_content=audio_content,
            audio_filename=audio_file.filename or "audio.mp3",
            topic=topic
        )
        
        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=result,
            message="Phân tích bài nói thành công"
        )
    
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi xử lý speech-to-text: {str(e)}"
        )


@router.post(
    "/text-to-speech",
    status_code=status.HTTP_200_OK,
)
async def text_to_speech(
    request: TextToSpeechRequest,
    background_tasks: BackgroundTasks
) -> FileResponse:
    """
    API chuyển đổi text to speech sử dụng gTTS
    
    - Nhận text từ người dùng
    - Sử dụng gTTS để chuyển đổi thành audio
    - Trả về file audio MP3
    """
    try:
        # Gọi service để xử lý
        tmp_file_path, file_response = text_to_speech_service(
            text=request.text,
            lang=request.lang
        )
        
        # Thêm task để xóa file sau khi response được gửi
        background_tasks.add_task(cleanup_temp_file, tmp_file_path)
        
        return file_response
    
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi xử lý text-to-speech: {str(e)}"
        )


@router.post(
    "/ocr-reading-analysis",
    response_model=DataResponse[OCRReadingResponse],
    status_code=status.HTTP_200_OK,
)
async def ocr_reading_analysis(
    image_file: UploadFile = File(..., description="File ảnh bài viết Reading IELTS"),
    topic: str = Form(None, description="Đề bài/chủ đề bài viết (tùy chọn)")
) -> Any:
    """
    API OCR ảnh thành text và phân tích bài viết Reading IELTS
    
    - Nhận file ảnh từ người dùng
    - Sử dụng PaddleOCR để chuyển đổi ảnh thành text
    - Gửi text cho LLM để phân tích band điểm, điểm mạnh, điểm yếu
    """
    try:
        # Kiểm tra file ảnh
        if not image_file.content_type or not image_file.content_type.startswith('image/'):
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="File phải là định dạng ảnh (jpg, png, etc.)"
            )
        
        # Đọc file ảnh
        image_content = await image_file.read()
        
        # Gọi service để xử lý
        result = await ocr_reading_analysis_service(
            image_content=image_content,
            image_filename=image_file.filename or "image.jpg",
            topic=topic
        )
        
        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=result,
            message="Phân tích bài viết thành công"
        )
    
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi xử lý OCR: {str(e)}"
        )
