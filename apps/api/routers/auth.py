"""Auth router: register, login, me."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from packages.shared import get_db, db_models
from packages.shared.security import hash_password, verify_password, create_access_token, rate_limit_ok
from apps.api.auth import auth
from apps.api.schemas import UserCreate, UserLogin, Token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=Token, status_code=201)
def register(body: UserCreate, db: Session = Depends(get_db)):
    if not rate_limit_ok("register"):
        raise HTTPException(429, "Too many requests")
    exists = db.query(db_models.User).filter(
        (db_models.User.username == body.username)
    ).first()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists")
    user = db_models.User(
        username=body.username, email=body.email,
        password_hash=hash_password(body.password),
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.username)
    return Token(access_token=token, user_id=user.id, username=user.username, is_admin=user.is_admin)


@router.post("/login", response_model=Token)
def login(body: UserLogin, db: Session = Depends(get_db)):
    if not rate_limit_ok(f"login:{body.username}"):
        raise HTTPException(429, "Too many requests")
    user = db.query(db_models.User).filter(db_models.User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")
    token = create_access_token(user.id, user.username)
    return Token(access_token=token, user_id=user.id, username=user.username, is_admin=user.is_admin)


@router.get("/me")
def me(user: db_models.User = Depends(auth)):
    return {"id": user.id, "username": user.username, "email": user.email, "is_admin": user.is_admin}
