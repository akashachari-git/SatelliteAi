import json
import time
import secrets
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from pathlib import Path
from ..config import GOOGLE_CLIENT_ID, SESSION_EXPIRY_HOURS, STORAGE_DIR

SESSIONS_FILE = STORAGE_DIR / "sessions.json"

class AuthService:
    """
    Secure Google Identity Services / OAuth 2.0 verification and session manager.
    Verifies authentic Google ID tokens with Google's public tokeninfo endpoint,
    identifies users by their permanent Google 'sub' ID, and manages server sessions.
    """
    
    _sessions: Dict[str, Dict[str, Any]] = {}
    _loaded: bool = False

    @classmethod
    def _load_sessions(cls):
        if cls._loaded:
            return
        cls._loaded = True
        if SESSIONS_FILE.exists():
            try:
                with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    now = time.time()
                    # Filter out expired sessions
                    cls._sessions = {
                        k: v for k, v in raw.items()
                        if v.get("expires_at", 0) > now
                    }
            except Exception as e:
                print(f"Notice: Failed to load sessions cache: {e}")
                cls._sessions = {}

    @classmethod
    def _save_sessions(cls):
        try:
            with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(cls._sessions, f, indent=2)
        except Exception as e:
            print(f"Notice: Failed to save sessions cache: {e}")

    @classmethod
    def verify_google_id_token(cls, id_token: str) -> Dict[str, Any]:
        """
        Cryptographically verifies the Google ID token directly with Google's
        official OAuth2 tokeninfo service.
        Guarantees that:
        1. The token was signed by Google
        2. The token is not expired
        3. The email is verified by Google
        4. The audience matches GOOGLE_CLIENT_ID (if configured)
        """
        if not id_token or not id_token.strip():
            raise ValueError("Missing Google ID token")

        token_clean = id_token.strip()
        url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token_clean}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SatQuery-AI-Auth/1.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            print(f"Google Token Verification HTTP Error {e.code}: {err_body}")
            raise ValueError("Google sign-in failed. Invalid or expired token.")
        except Exception as e:
            print(f"Google Token Verification Connection Error: {e}")
            raise ValueError("Google sign-in failed. Could not reach Google authentication services.")

        # Validate issuer
        iss = payload.get("iss", "")
        if iss not in ["accounts.google.com", "https://accounts.google.com"]:
            raise ValueError("Google sign-in failed. Unrecognized token issuer.")

        # Validate Google sub (Subject identifier)
        sub = payload.get("sub")
        if not sub:
            raise ValueError("Google sign-in failed. Token missing subject identifier.")

        # Validate audience if client ID is configured
        if GOOGLE_CLIENT_ID:
            aud = payload.get("aud", "")
            if aud != GOOGLE_CLIENT_ID:
                raise ValueError("Google sign-in failed. Client ID mismatch.")

        # Validate email
        email = payload.get("email")
        if not email:
            raise ValueError("Google sign-in failed. Email address not provided.")

        email_verified = payload.get("email_verified")
        if email_verified is False or str(email_verified).lower() == "false":
            raise ValueError("Google sign-in failed. Google email is not verified.")

        # Construct verified user profile
        user_profile = {
            "id": sub,
            "sub": sub,
            "email": email,
            "name": payload.get("name") or email.split("@")[0],
            "picture": payload.get("picture", ""),
            "given_name": payload.get("given_name", ""),
            "family_name": payload.get("family_name", ""),
            "auth_provider": "google",
            "verified_at": int(time.time())
        }

        return user_profile

    @classmethod
    def create_session(cls, user_profile: Dict[str, Any]) -> str:
        """Creates a secure server-side session for an authenticated user."""
        cls._load_sessions()
        session_token = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + (SESSION_EXPIRY_HOURS * 3600)

        cls._sessions[session_token] = {
            "user": user_profile,
            "created_at": now,
            "expires_at": expires_at
        }
        cls._save_sessions()
        return session_token

    @classmethod
    def get_session_user(cls, session_token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Retrieves and validates the active user session."""
        if not session_token:
            return None
        cls._load_sessions()
        session = cls._sessions.get(session_token)
        if not session:
            return None

        if session.get("expires_at", 0) < time.time():
            # Session expired
            cls.destroy_session(session_token)
            return None

        return session.get("user")

    @classmethod
    def destroy_session(cls, session_token: Optional[str]) -> bool:
        """Destroys an active session on logout."""
        if not session_token:
            return False
        cls._load_sessions()
        if session_token in cls._sessions:
            del cls._sessions[session_token]
            cls._save_sessions()
            return True
        return False
