package db

import (
	"context"
	"embed"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed ../../migrations/*.sql
var migrations embed.FS

// RunMigrations applies all embedded SQL files in order.
// For production, replace with golang-migrate or atlas.
func RunMigrations(ctx context.Context, pool *pgxpool.Pool) error {
	entries, err := migrations.ReadDir("../../migrations")
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}
	for _, e := range entries {
		sql, err := migrations.ReadFile("../../migrations/" + e.Name())
		if err != nil {
			return fmt.Errorf("read %s: %w", e.Name(), err)
		}
		if _, err := pool.Exec(ctx, string(sql)); err != nil {
			return fmt.Errorf("exec %s: %w", e.Name(), err)
		}
	}
	return nil
}
