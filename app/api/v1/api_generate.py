from typing import Any
from fastapi import APIRouter, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse

from app.utils.exception_handler import CustomException
from app.schemas.sche_response import DataResponse
from app.schemas.sche_generate import (
    GenerateSentenceRequest,
    GenerateSentenceResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    OCRReadingResponse,
    ChatBotRequest,
    ChatBotResponse,
)
from app.services.srv_generate import (
    generate_sentence_service,
    speech_to_text_analysis_service,
    text_to_speech_service,
    ocr_reading_analysis_service,
    cleanup_temp_file,
    chat_bot_service,
    chat_bot_stream_generator,
)

router = APIRouter(prefix=f"/generate")


@router.post(
    "/generate-sentence",
    response_model=DataResponse[GenerateSentenceResponse],
    status_code=status.HTTP_200_OK,
)
def generate_sentence(request: GenerateSentenceRequest) -> Any:

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
async def writing_analysis(
    image_file: UploadFile = File(None, description="File ảnh bài viết IELTS (truyền ảnh HOẶC text, không truyền cả 2)"),
    text: str = Form(None, description="Text bài viết trực tiếp (truyền ảnh HOẶC text, không truyền cả 2)"),
    topic: str = Form(None, description="Đề bài/chủ đề bài viết (tùy chọn)")
) -> Any:
    try:
        has_image = image_file is not None and image_file.filename
        has_text = text is not None and text.strip()
        
        # Validate: chỉ được truyền 1 trong 2
        if has_image and has_text:
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="Chỉ được truyền ảnh HOẶC text, không được truyền cả 2"
            )
        
        if not has_image and not has_text:
            raise CustomException(
                http_code=status.HTTP_400_BAD_REQUEST,
                message="Phải truyền ảnh hoặc text bài viết"
            )
        
        # Nếu có ảnh -> OCR rồi phân tích
        if has_image:
            # Kiểm tra file ảnh
            if not image_file.content_type or not image_file.content_type.startswith('image/'):
                raise CustomException(
                    http_code=status.HTTP_400_BAD_REQUEST,
                    message="File phải là định dạng ảnh (jpg, png, etc.)"
                )
            
            # Đọc file ảnh
            image_content = await image_file.read()
            
            # Gọi service để xử lý OCR + phân tích
            result = await ocr_reading_analysis_service(
                image_content=image_content,
                image_filename=image_file.filename or "image.jpg",
                topic=topic
            )
        else:
            # Nếu có text -> phân tích trực tiếp
            result = await ocr_reading_analysis_service(
                text_content=text.strip(),
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
            message=f"Lỗi khi xử lý phân tích bài viết: {str(e)}"
        )


@router.post(
    "/chat-bot",
    response_model=DataResponse[ChatBotResponse],
    status_code=status.HTTP_200_OK,
)
def chat_bot(request: ChatBotRequest) -> Any:
    try:
        result = chat_bot_service(
            question=request.question,
            history=request.history,
        )

        return DataResponse(
            http_code=status.HTTP_200_OK,
            data=result,
            message="Chatbot trả lời thành công",
        )
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi xử lý chatbot: {str(e)}"
        )


@router.post(
    "/chat-bot-stream",
    status_code=status.HTTP_200_OK,
)
def chat_bot_stream(request: ChatBotRequest) -> StreamingResponse:
    """
    Endpoint trả về câu trả lời chatbot dạng streaming (plain text).
    """
    try:
        generator = chat_bot_stream_generator(
            question=request.question,
            history=request.history,
        )
        return StreamingResponse(generator, media_type="text/plain; charset=utf-8")
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(
            http_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=f"Lỗi khi xử lý chatbot streaming: {str(e)}"
        )
