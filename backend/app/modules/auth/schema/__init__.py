"""
인증 모듈 스키마
"""

from pydantic import BaseModel, EmailStr, Field

# --- 요청 스키마 ---


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    display_name: str | None = Field(None, max_length=100)
    role: str = Field(default="user", pattern=r"^(user|lawyer)$")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=100)


class ProfileUpdateRequest(BaseModel):
    display_name: str | None = Field(None, max_length=100)
    avatar_url: str | None = Field(None, max_length=500)


class RoleUpdateRequest(BaseModel):
    role: str = Field(pattern=r"^(user|lawyer)$")


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=100)


# --- 응답 스키마 ---


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str | None
    avatar_url: str | None
    role: str
    email_verified: bool

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    user: UserResponse
    message: str = "success"


class MessageResponse(BaseModel):
    message: str
