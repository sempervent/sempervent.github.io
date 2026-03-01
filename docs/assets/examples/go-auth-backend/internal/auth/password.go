package auth

import "golang.org/x/crypto/bcrypt"

// Hash returns the bcrypt hash of the plaintext password.
func Hash(password string, cost int) (string, error) {
	b, err := bcrypt.GenerateFromPassword([]byte(password), cost)
	return string(b), err
}

// Check returns nil if password matches hash, bcrypt.ErrMismatchedHashAndPassword otherwise.
func Check(hash, password string) error {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
}
