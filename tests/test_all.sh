#!/bin/bash
# test_all.sh — integration tests for all molequla elements (build + smoke); the Go suite covers the witness and the field
# Can be run from anywhere: auto-detects repo root.
set -euo pipefail

# cd to repo root (one level up from tests/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH=$PATH:/usr/local/go/bin:$HOME/.cargo/bin

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0
SKIP=0

pass() { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}FAIL${NC} $1: $2"; FAIL=$((FAIL+1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC} $1: $2"; SKIP=$((SKIP+1)); }

TESTDIR=$(mktemp -d /tmp/molequla_test_XXXX)
cd ~/molequla

echo "═══════════════════════════════════════════════════"
echo " molequla integration tests"
echo " $(date)"
echo " testdir: $TESTDIR"
echo "═══════════════════════════════════════════════════"
echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}1. BUILD${NC}"
echo "─────────────────────────────────────────────────"

# Go (must build from module dir)
if go build -o $TESTDIR/molequla_go molequla.go 2>/dev/null; then
    pass "go build"
else
    fail "go build" "compilation failed"
fi

# C
if gcc -O2 -Wno-format-truncation -o $TESTDIR/molequla_c molequla.c -lm -lsqlite3 -lpthread 2>/dev/null; then
    pass "c build"
else
    fail "c build" "compilation failed"
fi

# Rust
if cargo build --release 2>/dev/null; then
    cp target/release/molequla $TESTDIR/molequla_rs 2>/dev/null || true
    pass "rust build"
else
    fail "rust build" "compilation failed"
fi

# JS — verify syntax
if node -c molequla.js 2>/dev/null; then
    pass "js syntax check"
else
    fail "js syntax" "parse error"
fi

echo ""

# ─────────────────────────────────────────────────────
echo -e "${CYAN}2. ELEMENT SMOKE TESTS${NC}"
echo "─────────────────────────────────────────────────"

cp nonames.txt $TESTDIR/

# Go: writes to stdout "[init] Stage 0..."
echo -n "  "
cd $TESTDIR
timeout 10 ./molequla_go > go.log 2>&1 || true
if grep -qi "stage\|init\|train\|step" go.log 2>/dev/null; then
    GOLINES=$(wc -l < go.log)
    pass "go smoke ($GOLINES lines, starts training)"
elif [ -f memory.sqlite3 ]; then
    pass "go smoke (created memory.sqlite3)"
else
    fail "go smoke" "no output and no db"
fi
cd ~/molequla

# C: writes to memory.sqlite3, stdout is quiet
echo -n "  "
cd $TESTDIR
rm -f memory.sqlite3
timeout 10 ./molequla_c > c.log 2>&1 || true
if [ -f memory.sqlite3 ]; then
    # SQLite files begin with the 16-byte header "SQLite format 3\0"
    if head -c 15 memory.sqlite3 | grep -q "SQLite format 3"; then
        pass "c smoke (memory.sqlite3 created, valid SQLite header)"
    else
        fail "c smoke" "memory.sqlite3 is not an SQLite file"
    fi
elif grep -qi "stage\|init\|train\|step\|corpus" c.log 2>/dev/null; then
    pass "c smoke (output detected)"
else
    fail "c smoke" "no db and no output"
fi
cd ~/molequla

# JS: browser-first, in Node exports modules
echo -n "  "
JSTEST=$(node -e "
const m = require('./molequla.js');
const checks = [];
if (m.GPT) checks.push('GPT');
if (m.EvolvingTokenizer) checks.push('Tokenizer');
if (m.DeltaAdapter) checks.push('DeltaAdapter');
if (m.SyntropyTracker) checks.push('SyntropyTracker');
if (m.SwarmRegistry) checks.push('SwarmRegistry');
console.log(checks.join(','));
" 2>/dev/null)
if echo "$JSTEST" | grep -q "GPT"; then
    pass "js smoke (exports: $JSTEST)"
else
    fail "js smoke" "module exports missing"
fi

# JS: verify key classes instantiate
JSINST=$(node -e "
const m = require('./molequla.js');
try {
    const tok = new m.EvolvingTokenizer();
    const tracker = new m.SyntropyTracker(4);
    const has_encode = typeof tok.encode === 'function';
    const has_decode = typeof tok.decode === 'function';
    console.log('PASS encode=' + has_encode + ' decode=' + has_decode);
} catch(e) { console.log('FAIL ' + e.message); }
" 2>/dev/null)
if echo "$JSINST" | grep -q "PASS"; then
    pass "js instantiation ($JSINST)"
else
    fail "js instantiation" "$JSINST"
fi

# Rust: outputs banner/phases
echo -n "  "
cd $TESTDIR
timeout 12 ./molequla_rs > rs.log 2>&1 || true
if grep -qi "element\|phase\|stage\|init\|Fourth\|molequla" rs.log 2>/dev/null; then
    RSLINES=$(wc -l < rs.log)
    pass "rust smoke ($RSLINES lines)"
elif [ -f memory.sqlite3 ]; then
    pass "rust smoke (created memory.sqlite3)"
else
    fail "rust smoke" "no output"
fi
cd ~/molequla

echo ""
echo ""
echo "═══════════════════════════════════════════════════"
echo -e " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "═══════════════════════════════════════════════════"

# Cleanup
rm -rf $TESTDIR

exit $FAIL
