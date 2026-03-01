package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sempervent/go-auth-backend/internal/config"
	appdb "github.com/sempervent/go-auth-backend/internal/db"
	apphttp "github.com/sempervent/go-auth-backend/internal/http"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "config error: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()

	pool, err := appdb.Connect(ctx, cfg.DatabaseURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "db connect: %v\n", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := appdb.RunMigrations(ctx, pool); err != nil {
		fmt.Fprintf(os.Stderr, "migrations: %v\n", err)
		os.Exit(1)
	}

	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      apphttp.NewServer(cfg, pool).Routes(),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Printf("listening on :%s", cfg.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("shutting down…")

	ctx2, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx2)
}
