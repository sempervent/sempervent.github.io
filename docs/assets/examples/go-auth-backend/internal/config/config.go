package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port          string
	DatabaseURL   string
	JWTSecret     string
	JWTTTLSeconds int
	BcryptCost    int
	CORSOrigins   string
}

func Load() (*Config, error) {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		return nil, fmt.Errorf("JWT_SECRET is required")
	}
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		return nil, fmt.Errorf("DATABASE_URL is required")
	}

	port := getEnv("APP_PORT", "8080")
	ttl := getEnvInt("JWT_TTL_SECONDS", 3600)
	cost := getEnvInt("BCRYPT_COST", 12)

	return &Config{
		Port:          port,
		DatabaseURL:   dbURL,
		JWTSecret:     secret,
		JWTTTLSeconds: ttl,
		BcryptCost:    cost,
		CORSOrigins:   os.Getenv("CORS_ORIGINS"),
	}, nil
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
