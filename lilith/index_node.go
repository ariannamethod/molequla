// index_node.go — Lilith INDEX node scaffolding
//
// INDEX node is a Go binary that:
// 1. Listens on a named pipe for commands from lilith.aml
// 2. Crawls Reddit (and later RSS, Wikipedia, Git repos)
// 3. Embeds text with a mini transformer (33M params)
// 4. Indexes into shared SQLite (sqlite-vec + FTS5)
// 5. Responds to lilith.aml with status/results
//
// This is the Go side of the named pipe IPC.
// The AML side is in ariannamethod.ai/core/ariannamethod.c (PIPE/INDEX commands).
//
// "Та, которая была до Евы."

package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"syscall"
)

// IndexNode represents a single INDEX crawling node
type IndexNode struct {
	ID      int
	CmdPipe string // path to command FIFO (read by this node)
	RspPipe string // path to response FIFO (write by this node)
	cmdFd   *os.File
	rspFd   *os.File
	running bool
}

// NewIndexNode creates a new INDEX node with standard pipe paths
func NewIndexNode(id int) *IndexNode {
	return &IndexNode{
		ID:      id,
		CmdPipe: fmt.Sprintf("/tmp/lilith_idx%d_cmd", id),
		RspPipe: fmt.Sprintf("/tmp/lilith_idx%d_rsp", id),
	}
}

// CreatePipes creates the named pipes (FIFOs) if they don't exist
func (n *IndexNode) CreatePipes() error {
	for _, path := range []string{n.CmdPipe, n.RspPipe} {
		os.Remove(path) // idempotent
		if err := syscall.Mkfifo(path, 0666); err != nil {
			if !os.IsExist(err) {
				return fmt.Errorf("mkfifo(%s): %w", path, err)
			}
		}
	}
	return nil
}

// Open opens the named pipes for communication
// CmdPipe: opened for reading (this node receives commands)
// RspPipe: opened for writing (this node sends responses)
func (n *IndexNode) Open() error {
	var err error

	// Open command pipe for reading (blocks until writer connects)
	n.cmdFd, err = os.OpenFile(n.CmdPipe, os.O_RDONLY, 0)
	if err != nil {
		return fmt.Errorf("open cmd pipe: %w", err)
	}

	// Open response pipe for writing (blocks until reader connects)
	n.rspFd, err = os.OpenFile(n.RspPipe, os.O_WRONLY, 0)
	if err != nil {
		n.cmdFd.Close()
		return fmt.Errorf("open rsp pipe: %w", err)
	}

	n.running = true
	return nil
}

// Close closes the pipes
func (n *IndexNode) Close() {
	n.running = false
	if n.cmdFd != nil {
		n.cmdFd.Close()
	}
	if n.rspFd != nil {
		n.rspFd.Close()
	}
}

// Respond writes a response line back to lilith.aml
func (n *IndexNode) Respond(msg string) error {
	if n.rspFd == nil {
		return fmt.Errorf("response pipe not open")
	}
	_, err := fmt.Fprintf(n.rspFd, "%s\n", msg)
	return err
}

// Listen reads commands from the command pipe and dispatches them
func (n *IndexNode) Listen() {
	scanner := bufio.NewScanner(n.cmdFd)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		log.Printf("[INDEX %d] received: %s", n.ID, line)

		parts := strings.SplitN(line, " ", 2)
		cmd := strings.ToUpper(parts[0])
		arg := ""
		if len(parts) > 1 {
			arg = parts[1]
		}

		switch cmd {
		case "FETCH":
			n.handleFetch(arg)
		case "STATUS":
			n.handleStatus()
		case "STOP":
			log.Printf("[INDEX %d] STOP received, shutting down", n.ID)
			n.Respond("OK stopped")
			n.running = false
			return
		default:
			log.Printf("[INDEX %d] unknown command: %s", n.ID, cmd)
			n.Respond(fmt.Sprintf("ERR unknown command: %s", cmd))
		}
	}
}

// handleFetch processes a FETCH command (e.g., "FETCH r/philosophy")
func (n *IndexNode) handleFetch(subreddit string) {
	// TODO: implement Reddit crawling
	// 1. Fetch posts from subreddit using Reddit .json endpoints
	// 2. Parse posts + top comments as Q&A pairs
	// 3. Embed with mini transformer (33M bge-small-en)
	// 4. Index into shared SQLite (sqlite-vec + FTS5)
	// 5. Convert to QA format for organism corpus
	log.Printf("[INDEX %d] FETCH %s — not yet implemented", n.ID, subreddit)
	n.Respond(fmt.Sprintf("OK queued %s", subreddit))
}

// handleStatus reports current INDEX node status
func (n *IndexNode) handleStatus() {
	// TODO: report actual crawl stats
	n.Respond(fmt.Sprintf("OK index_%d running=true indexed=0 pending=0", n.ID))
}

// Cleanup removes the named pipes from filesystem
func (n *IndexNode) Cleanup() {
	os.Remove(n.CmdPipe)
	os.Remove(n.RspPipe)
}
