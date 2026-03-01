package http

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/sempervent/go-auth-backend/internal/auth"
	"github.com/sempervent/go-auth-backend/internal/config"
)

// Server holds shared dependencies.
type Server struct {
	cfg  *config.Config
	pool *pgxpool.Pool

	// simple in-memory rate limiter for /auth/login
	mu       sync.Mutex
	attempts map[string]loginBucket
}

type loginBucket struct {
	count     int
	resetAt   time.Time
}

func NewServer(cfg *config.Config, pool *pgxpool.Pool) *Server {
	return &Server{cfg: cfg, pool: pool, attempts: make(map[string]loginBucket)}
}

func (s *Server) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("POST /auth/register", s.handleRegister)
	mux.HandleFunc("POST /auth/login", s.handleLogin)
	mux.Handle("GET /me", s.requireAuth(http.HandlerFunc(s.handleMe)))
	return mux
}

// ── helpers ───────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func readJSON(r *http.Request, v any) error {
	d := json.NewDecoder(r.Body)
	d.DisallowUnknownFields()
	return d.Decode(v)
}

// ── /healthz ─────────────────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// ── /auth/register ────────────────────────────────────────────────────────────

type registerReq struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	var req registerReq
	if err := readJSON(r, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request"})
		return
	}
	if req.Email == "" || req.Password == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "email and password are required"})
		return
	}
	if len(req.Password) < 8 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "password must be at least 8 characters"})
		return
	}

	hash, err := auth.Hash(req.Password, s.cfg.BcryptCost)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var id string
	err = s.pool.QueryRow(ctx,
		`INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id`,
		strings.ToLower(req.Email), hash,
	).Scan(&id)
	if err != nil {
		// Don't leak whether email already exists
		if strings.Contains(err.Error(), "unique") {
			writeJSON(w, http.StatusConflict, map[string]string{"error": "registration failed"})
			return
		}
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
		return
	}

	writeJSON(w, http.StatusCreated, map[string]string{"id": id})
}

// ── /auth/login ───────────────────────────────────────────────────────────────

type loginReq struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	// Rate limit: max 10 attempts per IP per minute
	ip := r.RemoteAddr
	if i := strings.LastIndex(ip, ":"); i != -1 {
		ip = ip[:i]
	}
	s.mu.Lock()
	b := s.attempts[ip]
	if time.Now().After(b.resetAt) {
		b = loginBucket{resetAt: time.Now().Add(time.Minute)}
	}
	b.count++
	s.attempts[ip] = b
	s.mu.Unlock()
	if b.count > 10 {
		writeJSON(w, http.StatusTooManyRequests, map[string]string{"error": "too many requests"})
		return
	}

	var req loginReq
	if err := readJSON(r, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	var (
		userID       string
		passwordHash string
	)
	err := s.pool.QueryRow(ctx,
		`SELECT id, password_hash FROM users WHERE email = $1`,
		strings.ToLower(req.Email),
	).Scan(&userID, &passwordHash)

	// Constant-time failure: always check bcrypt even on not-found to avoid timing attacks
	checkHash := passwordHash
	if err != nil {
		checkHash = "$2a$12$invalidhashfortimingXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	}
	checkErr := auth.Check(checkHash, req.Password)
	if err != nil || checkErr != nil {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid credentials"})
		return
	}

	token, err := auth.Issue(userID, req.Email, s.cfg.JWTSecret, s.cfg.JWTTTLSeconds)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"token": token})
}

// ── /me ───────────────────────────────────────────────────────────────────────

type userKey struct{}

func (s *Server) requireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hdr := r.Header.Get("Authorization")
		if !strings.HasPrefix(hdr, "Bearer ") {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "missing token"})
			return
		}
		claims, err := auth.Verify(strings.TrimPrefix(hdr, "Bearer "), s.cfg.JWTSecret)
		if err != nil {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid token"})
			return
		}
		ctx := context.WithValue(r.Context(), userKey{}, claims)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (s *Server) handleMe(w http.ResponseWriter, r *http.Request) {
	claims := r.Context().Value(userKey{}).(*auth.Claims)
	writeJSON(w, http.StatusOK, map[string]string{
		"id":    claims.Subject,
		"email": claims.Email,
	})
}
