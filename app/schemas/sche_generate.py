from typing import Optional, List

from pydantic import BaseModel, Field


class GenerateSentenceRequest(BaseModel):
    vocabulary: str = Field(..., description="Từ vựng cần sinh câu", min_length=1)


class GenerateSentenceResponse(BaseModel):
    sentence: str = Field(..., description="Câu được sinh ra chứa từ vựng")
    vocabulary: str = Field(..., description="Từ vựng đã được sử dụng trong câu")


class SpeechToTextResponse(BaseModel):
    transcript: str = Field(..., description="Nội dung text được chuyển đổi từ audio")
    band_score: str = Field(..., description="Band điểm IELTS dự kiến")
    strengths: str = Field(..., description="Điểm mạnh trong bài nói")
    weaknesses: str = Field(..., description="Điểm yếu cần cải thiện")
    overall_feedback: str = Field(..., description="Nhận xét tổng quan về bài nói")


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text cần chuyển đổi thành audio", min_length=1)
    lang: str = Field(default="en", description="Ngôn ngữ (mặc định: en - tiếng Anh)")


class OCRReadingResponse(BaseModel):
    extracted_text: str = Field(..., description="Text được trích xuất từ ảnh")
    band_score: str = Field(..., description="Band điểm IELTS Reading dự kiến")
    strengths: str = Field(..., description="Điểm mạnh trong bài viết")
    weaknesses: str = Field(..., description="Điểm yếu cần cải thiện")
    overall_feedback: str = Field(..., description="Nhận xét tổng quan về bài viết")
    suggestions: str = Field(..., description="Gợi ý cải thiện")


class ChatBotRequest(BaseModel):
    question: str = Field(
        ...,
        description="Câu hỏi/thắc mắc về tiếng Anh hoặc kỳ thi IELTS của học viên",
        min_length=1,
    )
    history: Optional[List[dict]] = Field(
        default=None,
        description="Lịch sử hội thoại trước đó, dạng list các object {role, content} với role thuộc {system,user,assistant}",
    )


class ChatBotResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Câu trả lời/gợi ý chi tiết cho câu hỏi của học viên, có ví dụ minh hoạ nếu phù hợp",
    )
