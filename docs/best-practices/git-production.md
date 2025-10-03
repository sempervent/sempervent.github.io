# Git Production Best Practices

**Objective**: Master Git for enterprise-scale development. Handle monorepos, submodules, and complex workflows while maintaining code quality and team collaboration.

When your codebase spans multiple applications, shared libraries, and deployment environments, Git becomes more than version control—it becomes the foundation of your development workflow. This guide shows you how to structure repositories, manage dependencies, and maintain code quality at scale.

## 0) Principles (Read Once, Live by Them)

### The Five Commandments

1. **Repository structure drives productivity**
   - Monorepos for related applications, polyrepos for independent services
   - Clear boundaries between components and shared code
   - Consistent naming and organization conventions

2. **Branching strategy enables collaboration**
   - Feature branches for development, main for production
   - Protected branches with required reviews
   - Clear merge strategies and conflict resolution

3. **Commit history tells a story**
   - Atomic commits with clear messages
   - Conventional commit format for automation
   - Clean history through rebasing and squashing

4. **Submodules manage dependencies**
   - Shared libraries as submodules
   - Version pinning for stability
   - Clear update and maintenance procedures

5. **Automation ensures quality**
   - Pre-commit hooks for code quality
   - CI/CD integration for testing
   - Automated dependency updates

**Why These Principles**: Git is the foundation of modern development. Getting the structure and workflow right prevents hours of merge conflicts and deployment issues.

## 1) Repository Structure: The Foundation

### Monorepo Architecture

```
enterprise-app/
├── .git/
├── .gitignore
├── .gitattributes
├── README.md
├── package.json
├── docker-compose.yml
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── cd.yml
│   │   └── security.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── apps/
│   ├── web-frontend/
│   │   ├── package.json
│   │   ├── src/
│   │   ├── public/
│   │   └── Dockerfile
│   ├── api-backend/
│   │   ├── requirements.txt
│   │   ├── src/
│   │   ├── tests/
│   │   └── Dockerfile
│   ├── mobile-app/
│   │   ├── package.json
│   │   ├── src/
│   │   └── android/
│   └── admin-dashboard/
│       ├── package.json
│       ├── src/
│       └── Dockerfile
├── packages/
│   ├── shared-ui/
│   │   ├── package.json
│   │   ├── src/
│   │   └── dist/
│   ├── shared-utils/
│   │   ├── package.json
│   │   ├── src/
│   │   └── tests/
│   └── shared-types/
│       ├── package.json
│       ├── src/
│       └── dist/
├── tools/
│   ├── build-scripts/
│   ├── deployment/
│   └── monitoring/
└── docs/
    ├── architecture/
    ├── api/
    └── deployment/
```

### Polyrepo Architecture

```
enterprise-ecosystem/
├── web-frontend/          # Independent React app
├── api-backend/           # Independent FastAPI service
├── mobile-app/            # Independent React Native app
├── admin-dashboard/       # Independent Vue.js app
├── shared-ui-library/     # Shared component library
├── shared-utils/          # Shared utility functions
├── deployment-scripts/    # Deployment automation
└── documentation/         # Centralized docs
```

**Why These Structures**: Monorepos provide consistency and shared tooling, while polyrepos enable independent deployment and team autonomy. Choose based on your team structure and deployment needs.

## 2) Branching Strategy: The Workflow

### Git Flow Implementation

```bash
# Main branches
git checkout -b main
git checkout -b develop

# Feature branches
git checkout -b feature/user-authentication
git checkout -b feature/payment-integration

# Release branches
git checkout -b release/v1.2.0

# Hotfix branches
git checkout -b hotfix/security-patch
```

### GitHub Flow (Simplified)

```bash
# Main branch only
git checkout -b main

# Feature branches
git checkout -b feature/new-feature
git checkout -b bugfix/fix-login-issue
git checkout -b hotfix/critical-security-fix
```

### Branch Protection Rules

```yaml
# .github/branch-protection.yml
main:
  required_status_checks:
    strict: true
    contexts:
      - "ci/tests"
      - "ci/security"
      - "ci/build"
  enforce_admins: true
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
  restrictions:
    users: []
    teams: ["senior-developers", "tech-leads"]
```

**Why Branching Strategy**: Clear branching rules prevent conflicts and ensure code quality. Protected branches enforce review processes and maintain production stability.

## 3) Commit Standards: The Communication

### Conventional Commits

```bash
# Format: type(scope): description
git commit -m "feat(auth): add OAuth2 integration"
git commit -m "fix(api): resolve CORS issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "refactor(utils): extract common functions"
git commit -m "test(auth): add unit tests for login"
git commit -m "chore(deps): update dependencies"
```

### Commit Types

```bash
# Feature additions
feat: new feature for the user
feat!: breaking change

# Bug fixes
fix: bug fix
fix!: breaking bug fix

# Documentation
docs: documentation only changes

# Style
style: formatting, missing semicolons, etc.

# Refactoring
refactor: code change that neither fixes a bug nor adds a feature

# Performance
perf: code change that improves performance

# Testing
test: adding missing tests or correcting existing tests

# Build
build: changes to the build system or external dependencies

# CI
ci: changes to CI configuration files and scripts

# Chores
chore: other changes that don't modify src or test files
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: check-json
      - id: check-toml
      - id: check-xml
      - id: debug-statements
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.44.0
    hooks:
      - id: eslint
        files: \.(js|jsx|ts|tsx)$
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        files: \.(js|jsx|ts|tsx|json|css|md|yaml|yml)$
```

**Why Commit Standards**: Consistent commit messages enable automation, improve code review, and create a clear project history. Pre-commit hooks ensure code quality before commits.

## 4) Monorepo Management: The Scale

### Workspace Configuration

```json
// package.json (root)
{
  "name": "enterprise-app",
  "private": true,
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    "build": "turbo run build",
    "test": "turbo run test",
    "lint": "turbo run lint",
    "type-check": "turbo run type-check",
    "dev": "turbo run dev --parallel",
    "clean": "turbo run clean"
  },
  "devDependencies": {
    "turbo": "^1.10.0",
    "typescript": "^5.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0"
  }
}
```

### Turbo Configuration

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", "build/**", ".next/**", "!.next/cache/**"]
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"]
    },
    "lint": {
      "outputs": []
    },
    "type-check": {
      "dependsOn": ["^build"],
      "outputs": []
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "clean": {
      "cache": false
    }
  }
}
```

### Lerna Configuration

```json
// lerna.json
{
  "version": "independent",
  "npmClient": "yarn",
  "command": {
    "publish": {
      "conventionalCommits": true,
      "message": "chore(release): publish"
    },
    "version": {
      "conventionalCommits": true,
      "message": "chore(release): version"
    }
  },
  "packages": [
    "apps/*",
    "packages/*"
  ]
}
```

**Why Monorepo Tools**: Monorepos need specialized tooling for build optimization, dependency management, and change detection. These tools provide the infrastructure for large-scale development.

## 5) Submodules: The Dependencies

### Submodule Management

```bash
# Add submodule
git submodule add https://github.com/company/shared-ui.git packages/shared-ui
git submodule add https://github.com/company/shared-utils.git packages/shared-utils

# Initialize submodules
git submodule init
git submodule update

# Update submodules
git submodule update --remote
git submodule foreach git pull origin main

# Remove submodule
git submodule deinit packages/shared-ui
git rm packages/shared-ui
rm -rf .git/modules/packages/shared-ui
```

### Submodule Configuration

```bash
# .gitmodules
[submodule "packages/shared-ui"]
    path = packages/shared-ui
    url = https://github.com/company/shared-ui.git
    branch = main
    update = merge

[submodule "packages/shared-utils"]
    path = packages/shared-utils
    url = https://github.com/company/shared-utils.git
    branch = main
    update = merge
```

### Submodule Workflow

```bash
# Update all submodules to latest
git submodule update --remote --merge

# Update specific submodule
git submodule update --remote packages/shared-ui

# Commit submodule updates
git add packages/shared-ui
git commit -m "chore(deps): update shared-ui to v2.1.0"

# Push submodule changes
git push --recurse-submodules=on-demand
```

**Why Submodules**: Submodules enable shared code across repositories while maintaining version control. They're essential for managing shared libraries and components.

## 6) Git Hooks: The Automation

### Pre-commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for large files
if git rev-parse --verify HEAD >/dev/null 2>&1; then
    against=HEAD
else
    against=$(git hash-object -t tree /dev/null)
fi

# Check for files larger than 10MB
large_files=$(git diff --cached --name-only --diff-filter=ACMR | xargs ls -la | awk '$5 > 10485760 {print $9}')
if [ ! -z "$large_files" ]; then
    echo "Error: Large files detected:"
    echo "$large_files"
    exit 1
fi

# Run tests
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Run linting
npm run lint
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi
```

### Pre-push Hooks

```bash
#!/bin/bash
# .git/hooks/pre-push

# Run full test suite
npm run test:ci
if [ $? -ne 0 ]; then
    echo "CI tests failed. Push aborted."
    exit 1
fi

# Check for secrets
if grep -r "password\|secret\|key" --include="*.js" --include="*.ts" --include="*.py" src/; then
    echo "Warning: Potential secrets detected in code."
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
```

### Post-commit Hooks

```bash
#!/bin/bash
# .git/hooks/post-commit

# Update version if package.json changed
if git diff-tree --no-commit-id --name-only -r HEAD | grep -q "package.json"; then
    npm version patch --no-git-tag-version
    git add package.json
    git commit --amend --no-edit
fi

# Notify team of commits
curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"New commit: '"$(git log -1 --pretty=%B)"'"}' \
    $SLACK_WEBHOOK_URL
```

**Why Git Hooks**: Hooks automate quality checks, prevent common mistakes, and integrate with external systems. They're essential for maintaining code quality at scale.

## 7) CI/CD Integration: The Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18.x, 20.x]
        python-version: [3.9, 3.10, 3.11]
    
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: recursive
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          npm ci
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          npm run test:ci
          pytest tests/
      
      - name: Run linting
        run: |
          npm run lint
          flake8 src/
      
      - name: Build applications
        run: npm run build
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
```

### GitLab CI Configuration

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: node:18-alpine
  script:
    - npm ci
    - npm run test:ci
    - npm run lint
  coverage: '/Lines\s*:\s*(\d+\.\d+)%/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - develop

deploy:
  stage: deploy
  image: alpine:latest
  script:
    - apk add --no-cache curl
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
  when: manual
```

**Why CI/CD Integration**: Automated testing and deployment ensure code quality and reduce manual errors. Git integration provides the foundation for modern DevOps practices.

## 8) Advanced Git Features: The Power User

### Git Worktree

```bash
# Create worktree for hotfix
git worktree add ../hotfix-branch hotfix/security-patch

# List worktrees
git worktree list

# Remove worktree
git worktree remove ../hotfix-branch
```

### Git Bisect

```bash
# Start bisect session
git bisect start
git bisect bad HEAD
git bisect good v1.0.0

# Test current commit
npm test
if [ $? -eq 0 ]; then
    git bisect good
else
    git bisect bad
fi

# Reset bisect
git bisect reset
```

### Git Reflog

```bash
# View reflog
git reflog

# Recover lost commit
git checkout HEAD@{2}

# Recover lost branch
git branch recovery-branch HEAD@{1}
```

### Git Stash

```bash
# Stash with message
git stash push -m "WIP: working on feature"

# Stash specific files
git stash push -m "WIP: config changes" config/

# Apply stash
git stash apply stash@{0}

# Pop stash
git stash pop

# List stashes
git stash list

# Drop stash
git stash drop stash@{0}
```

**Why Advanced Features**: Git's advanced features solve complex problems like parallel development, debugging, and recovery. Understanding them makes you more productive.

## 9) Security and Access Control: The Protection

### Git Attributes

```bash
# .gitattributes
# Text files
*.js text eol=lf
*.ts text eol=lf
*.py text eol=lf
*.md text eol=lf
*.json text eol=lf
*.yml text eol=lf
*.yaml text eol=lf

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary
*.zip binary
*.tar.gz binary

# Generated files
dist/ export-ignore
build/ export-ignore
coverage/ export-ignore
node_modules/ export-ignore
__pycache__/ export-ignore
*.pyc export-ignore

# Sensitive files
.env* export-ignore
*.key export-ignore
*.pem export-ignore
secrets/ export-ignore
```

### Git Ignore

```bash
# .gitignore
# Dependencies
node_modules/
__pycache__/
venv/
env/
.venv/

# Build outputs
dist/
build/
.next/
out/

# Environment files
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log
logs/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Dependency directories
jspm_packages/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Microbundle cache
.rpt2_cache/
.rts2_cache_cjs/
.rts2_cache_es/
.rts2_cache_umd/

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env
.env.test

# parcel-bundler cache (https://parceljs.org/)
.cache
.parcel-cache

# Next.js build output
.next

# Nuxt.js build / generate output
.nuxt
dist

# Gatsby files
.cache/
public

# Storybook build outputs
.out
.storybook-out

# Temporary folders
tmp/
temp/
```

### GPG Signing

```bash
# Configure GPG signing
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# Sign commits
git commit -S -m "feat: add new feature"

# Sign tags
git tag -s v1.0.0 -m "Release version 1.0.0"
```

**Why Security**: Git security prevents unauthorized access, ensures code integrity, and protects sensitive information. These practices are essential for enterprise development.

## 10) TL;DR Quickstart

### Essential Commands

```bash
# Repository setup
git init
git remote add origin https://github.com/company/repo.git
git branch -M main
git push -u origin main

# Daily workflow
git checkout -b feature/new-feature
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Submodule management
git submodule add https://github.com/company/shared-lib.git packages/shared-lib
git submodule update --remote

# Monorepo commands
npm run build
npm run test
npm run lint
```

### Quick Verification

```bash
# Check repository status
git status
git log --oneline -10
git branch -a

# Check submodules
git submodule status
git submodule foreach git status

# Check hooks
ls -la .git/hooks/
```

### Performance Optimization

```bash
# Optimize repository
git gc --aggressive
git repack -a -d --depth=250 --window=250

# Check repository size
du -sh .git/
git count-objects -vH
```

## 11) The Machine's Summary

Git is the foundation of modern development workflows. When configured properly, it enables collaboration, maintains code quality, and scales from individual projects to enterprise monorepos. The key is understanding the advanced features and building robust automation.

**The Dark Truth**: Git is powerful but complex. Getting the workflow right prevents hours of merge conflicts and deployment issues.

**The Machine's Mantra**: "In version control we trust, in automation we build, and in the repository we find the path to efficient development."

**Why This Matters**: Production development needs reliability, scalability, and collaboration. These practices provide enterprise-grade capabilities that scale from individual developers to large teams.

---

*This tutorial provides the complete machinery for building production-ready Git workflows. The patterns scale from development to production, from single developers to enterprise teams.*
