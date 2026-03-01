// Tiny static binary used as Docker HEALTHCHECK in the scratch image.
// Exits 0 if GET http://127.0.0.1:8080/healthz returns 2xx, else exits 1.
package main

import (
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("APP_PORT")
	if port == "" {
		port = "8080"
	}
	resp, err := http.Get("http://127.0.0.1:" + port + "/healthz")
	if err != nil || resp.StatusCode >= 300 {
		os.Exit(1)
	}
}
